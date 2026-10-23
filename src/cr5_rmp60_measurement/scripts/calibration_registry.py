#!/usr/bin/env python3
"""
校准注册表管理模块（Calibration Registry Manager）。

本模块负责加载、验证和查询校准注册表 —— 一个结构化的 JSON 数据库，
用于存储所有已校准的参数并确保其在各流程之间的一致性。

注册表 JSON 的结构：
{
  "schema_version": 1,                      # 模式版本，当前为 1
  "active_standard_sphere_id": "sphere_1",  # 当前激活的标准球 ID
  "standard_spheres": {                      # 所有已标定的标准球
    "<sphere_id>": {
      "sphere_center_mm": [x, y, z],        # 球心坐标（mm）
      "status": "accepted",                  # 标定状态
      "canonical_fit_json": "path/to/fit.json"  # 标准球的拟合结果 JSON 路径
    }
  },
  "styli": {                                 # 所有已标定的探针（触笔）
    "<stylus_id>": {
      "branch": "z",                         # 所属测量分支
      "sphere_id": "sphere_1",               # 标定所用的标准球 ID
      "local_ball_offset_mm": [x, y, z],     # 探针尖端到法兰的局部偏移
      "canonical_fit_json": "path/to/fit.json",   # 探针的拟合结果 JSON 路径
      "status": "accepted",                   # 标定状态
      "allowed_for_auto": true,               # 是否允许自动流程使用
      "fit_result_selector": {                # （可选）多分支拟合文件的选择器
        "branch": "z"                         # 从拟合结果中选取哪个分支的偏移量
      }
    }
  }
}

关键机制：
  - fit_result_selector（拟合结果选择器）：当一个拟合 JSON 文件包含多个分支的
    结果（branches 数组）时，此选择器指定从中选取哪个分支的 estimated_offset_mm。
    如果未设置选择器，则直接从拟合 JSON 顶层读取 local_ball_offset_mm。
  - 注册表是所有校准流程的"单一真相源"（single source of truth），所有下游
    操作（如机器人运动规划）都必须先通过注册表的验证，确保使用的参数一致。
"""
import argparse
import json
import math
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_DIR / "config/calibration_registry.json"


def project_path(value):
    """
    将给定的路径解析为相对于项目根目录的绝对路径。

    如果 value 是相对路径，则以项目根目录（PROJECT_DIR）为基准进行拼接；
    如果已是绝对路径，则直接返回。最终结果会被规范化（resolve）。

    Args:
        value: 路径字符串，可以是相对路径或绝对路径。

    Returns:
        解析后的绝对 Path 对象。
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def relative_project_path(value):
    """
    将绝对路径转换为相对于项目根目录的字符串表示。

    如果路径位于项目目录内，则返回相对路径（如 "config/foo.json"）；
    否则返回原始绝对路径。主要用于日志和错误消息的可读性。

    Args:
        value: 路径字符串或 Path 对象。

    Returns:
        简洁的路径字符串。
    """
    path = project_path(value)
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def load_registry(path=DEFAULT_REGISTRY):
    """
    加载并执行基本校验的校准注册表 JSON 文件。

    基本校验包括:
      1. schema_version 必须为 1（当前支持的版本）。
      2. active_standard_sphere_id 指向的标准球必须存在且状态为 "accepted"。

    更全面的交叉校验（如 sphere_center 一致性、stylus offset 一致性等）
    由 validate_registry() 完成。

    Args:
        path: 注册表 JSON 文件的路径（字符串或 Path）。

    Returns:
        解析后的注册表数据字典。

    Raises:
        OSError: 文件不存在或无法读取。
        ValueError: 注册表版本不兼容或激活的标准球缺失/未接受。
    """
    registry_path = project_path(path)
    data = json.loads(registry_path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{registry_path}: unsupported calibration registry schema")
    active_id = data.get("active_standard_sphere_id")
    sphere = data.get("standard_spheres", {}).get(active_id)
    if not active_id or not sphere or sphere.get("status") != "accepted":
        raise ValueError(f"{registry_path}: active accepted standard sphere is missing")
    return data


def parse_mapping(items, option_name):
    """
    解析命令行参数中的 BRANCH=VALUE 格式映射表。

    例如 ["z=stylus_z", "x_pos=stylus_xp"] 会被解析为
    {"z": "stylus_z", "x_pos": "stylus_xp"}。

    Args:
        items: 字符串列表，每个元素格式为 "BRANCH=VALUE"。
        option_name: 用于错误消息中的参数名称（如 "--branch-fit"）。

    Returns:
        映射字典 {branch: value}。

    Raises:
        ValueError: 格式错误、值为空或存在重复分支名。
    """
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{option_name} entries must be BRANCH=VALUE")
        branch, value = (part.strip() for part in item.split("=", 1))
        if not branch or not value:
            raise ValueError(f"{option_name} entries must be BRANCH=VALUE")
        if branch in result:
            raise ValueError(f"{option_name}: duplicate branch {branch}")
        result[branch] = value
    return result


def _vector(data, key, source):
    """从字典中提取并验证一个 3 元素向量。"""
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{source}: missing 3-value {key}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{source}: non-finite {key}")
    return result


def _assert_close(actual, expected, label, tolerance=1e-6):
    delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, expected)))
    if delta > tolerance:
        raise ValueError(f"{label} differs from registry by {delta:.6f} mm")


def _fit_offset(fit_data, stylus, source):
    """
    从拟合结果中提取 stylus 的局部偏移量（local_ball_offset_mm）。

    这个函数处理两种场景：
      1. 无 fit_result_selector：直接从拟合 JSON 顶层读取 local_ball_offset_mm。
      2. 有 fit_result_selector（如 {"branch": "z"}）：从拟合 JSON 的
         "branches" 数组中选取指定分支的 "estimated_offset_mm" 作为偏移量。
         这种机制适用于一个拟合文件同时包含多个分支测量结果的情况。

    Args:
        fit_data: 拟合结果的数据字典。
        stylus: 注册表中该 stylus 的条目，可能包含 fit_result_selector。
        source: 用于错误消息的来源描述（如文件路径）。

    Returns:
        3 元素偏移向量 [x, y, z]。
    """
    selector = stylus.get("fit_result_selector")
    if selector:
        branch = selector.get("branch")
        result = next(
            (item for item in fit_data.get("branches", []) if item.get("branch") == branch),
            None,
        )
        if result is None:
            raise ValueError(f"{source}: no fit result for selector branch {branch}")
        data = {"local_ball_offset_mm": result.get("estimated_offset_mm")}
    else:
        data = fit_data
    return _vector(data, "local_ball_offset_mm", source)


def validate_registry(registry):
    """
    执行注册表的完整交叉验证，确保所有参数在物理上自洽。

    验证逻辑包括：
      1. 标准球验证:
         - 检查 active_standard_sphere 的 canonical_fit_json 文件存在。
         - 验证拟合 JSON 中的 sphere_center_mm 与注册表中记录的值一致
           （默认容差 1e-6 mm）。

      2. 每个 stylus 验证:
         - 确认 stylus 所关联的 sphere_id 就是当前激活的标准球。
         - 检查 stylus 的 canonical_fit_json 文件存在。
         - 验证拟合文件的 reference_fit_json（如果存在）指向的是正确的标准球拟合文件。
         - 验证拟合 JSON 中的偏移量与注册表中记录的 local_ball_offset_mm 一致。
         - 如果拟合 JSON 中包含 sphere_center_mm，还需验证其与标准球的球心一致。

    这种双重重叠检查确保了：注册表中的数据与原始拟合结果文件之间
    没有发生静默偏差（silent drift），所有下游流程使用一致的参数。

    Args:
        registry: 由 load_registry() 加载的注册表字典。

    Returns:
        验证通过后的注册表字典。

    Raises:
        ValueError: 任何一致性检查失败时抛出，包含详细的失败原因。
    """
    active_id = registry["active_standard_sphere_id"]
    sphere = registry["standard_spheres"][active_id]
    sphere_fit_path = project_path(sphere["canonical_fit_json"])
    if not sphere_fit_path.is_file():
        raise ValueError(f"registered sphere fit does not exist: {sphere_fit_path}")
    sphere_fit = json.loads(sphere_fit_path.read_text())
    _assert_close(
        _vector(sphere_fit, "sphere_center_mm", sphere_fit_path),
        _vector(sphere, "sphere_center_mm", "registry sphere"),
        "canonical sphere centre",
    )

    for stylus_id, stylus in registry.get("styli", {}).items():
        if not stylus.get("allowed_for_auto", False):
            continue
        if stylus.get("sphere_id") != active_id:
            raise ValueError(f"{stylus_id}: sphere_id is not the active standard sphere")
        fit_path = project_path(stylus["canonical_fit_json"])
        if not fit_path.is_file():
            raise ValueError(f"{stylus_id}: registered fit does not exist: {fit_path}")
        fit_data = json.loads(fit_path.read_text())
        fit_reference = fit_data.get("reference_fit_json")
        if fit_reference is not None and project_path(fit_reference) != sphere_fit_path:
            raise ValueError(
                f"{stylus_id}: fit references a non-canonical sphere fit: {fit_reference}"
            )
        _assert_close(
            _fit_offset(fit_data, stylus, fit_path),
            _vector(stylus, "local_ball_offset_mm", stylus_id),
            f"{stylus_id} offset",
        )
        fit_center = fit_data.get("sphere_center_mm")
        if fit_center is not None:
            _assert_close(
                [float(value) for value in fit_center],
                _vector(sphere, "sphere_center_mm", "registry sphere"),
                f"{stylus_id} sphere centre",
            )
    return registry


def resolve_branch_styli(
    registry,
    branches,
    fits,
    explicit_mapping=None,
    reference_fit_json=None,
    require_auto_ready=False,
):
    """
    将测量分支（branch）映射到注册表中已注册的物理 stylus（触笔）身份。

    映射流程：
      1. 如果提供了 reference_fit_json，验证它是否与标准球的 canonical_fit 一致。
      2. 检查 explicit_mapping 中的分支是否都属于给定的 branches 列表。
      3. 检查废弃的拟合文件（deprecated_artifacts），已废弃的路径将被拒绝。
      4. 对每个分支按以下优先级确定 stylus_id：
         a) explicit_mapping 中明确指定的 stylus_id。
         b) 如果未指定，则根据 fit 路径匹配注册表中 branch + canonical_fit_json 都一致的条目。
      5. 对匹配到的 stylus 进行一系列约束检查：
         - stylus 存在。
         - stylus 注册的分支与当前分支一致。
         - stylus 关联的标准球是当前激活的标准球。
         - 选中的 fit 路径是 stylus 的 canonical_fit。
         - 如果 require_auto_ready，stylus 必须标记为 allowed_for_auto=True。

    Args:
        registry: 注册表字典。
        branches: 所有需要参与映射的分支名称列表。
        fits: {branch: fit_path} 字典，每个分支对应的拟合文件路径。
        explicit_mapping: 可选的 {branch: stylus_id} 显式映射。
        reference_fit_json: 标准球的参考拟合路径（用于额外验证）。
        require_auto_ready: 如果为 True，只允许 allowed_for_auto 的 stylus。

    Returns:
        {branch: stylus_id} 解析结果字典。

    Raises:
        ValueError: 任何分支无法解析或约束检查失败时抛出。
    """
    active_id = registry["active_standard_sphere_id"]
    sphere = registry["standard_spheres"][active_id]
    if reference_fit_json is not None:
        expected = project_path(sphere["canonical_fit_json"])
        actual = project_path(reference_fit_json)
        if actual != expected:
            raise ValueError(
                "reference sphere fit is not canonical: "
                f"{relative_project_path(actual)} != {relative_project_path(expected)}"
            )

    explicit_mapping = explicit_mapping or {}
    unknown = sorted(set(explicit_mapping) - set(branches))
    if unknown:
        raise ValueError("stylus mapping contains inactive branches: " + ", ".join(unknown))

    deprecated = {
        project_path(item["path"]): item
        for item in registry.get("deprecated_artifacts", [])
        if item.get("path")
    }
    styli = registry.get("styli", {})
    resolved = {}
    failures = []
    for branch in branches:
        fit = fits.get(branch)
        fit_path = project_path(fit) if fit is not None else None
        if fit_path in deprecated:
            item = deprecated[fit_path]
            failures.append(f"{branch}: deprecated fit ({item.get('status')}): {relative_project_path(fit_path)}")
            continue

        stylus_id = explicit_mapping.get(branch)
        if stylus_id is None and fit_path is not None:
            matches = [
                candidate_id
                for candidate_id, candidate in styli.items()
                if candidate.get("branch") == branch
                and project_path(candidate.get("canonical_fit_json")) == fit_path
            ]
            if len(matches) == 1:
                stylus_id = matches[0]
        if stylus_id is None:
            failures.append(f"{branch}: no unambiguous physical stylus identity for selected fit")
            continue
        stylus = styli.get(stylus_id)
        if stylus is None:
            failures.append(f"{branch}: unknown physical stylus id {stylus_id}")
            continue
        if stylus.get("branch") != branch:
            failures.append(
                f"{branch}: {stylus_id} is registered as branch {stylus.get('branch')}"
            )
            continue
        if stylus.get("sphere_id") != active_id:
            failures.append(f"{branch}: {stylus_id} belongs to another sphere setup")
            continue
        canonical_fit = project_path(stylus["canonical_fit_json"])
        if fit_path != canonical_fit:
            failures.append(
                f"{branch}: selected fit {relative_project_path(fit_path)} does not match "
                f"{stylus_id} canonical fit {relative_project_path(canonical_fit)}"
            )
            continue
        if require_auto_ready and not stylus.get("allowed_for_auto", False):
            failures.append(
                f"{branch}: {stylus_id} status={stylus.get('status')} is not auto-ready"
            )
            continue
        resolved[branch] = stylus_id
    if failures:
        raise ValueError("calibration identity gate failed: " + "; ".join(failures))
    return resolved


def parse_args():
    """
    解析命令行参数。

    CLI 参数：
      --registry             校准注册表 JSON 路径（默认 config/calibration_registry.json）。
      --reference-fit-json   标准球的参考拟合 JSON（用于额外校验）。
      --branches             需要解析的分支名称列表。
      --branch-fit           分支到拟合文件的映射，格式 BRANCH=JSON_PATH（可多次指定）。
      --branch-stylus        分支到 stylus_id 的映射，格式 BRANCH=STYLUS_ID（可多次指定）。
      --require-auto-ready   只允许标记为 allowed_for_auto 的 stylus。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--reference-fit-json")
    parser.add_argument("--branches", nargs="+", default=[])
    parser.add_argument("--branch-fit", nargs="*", default=[], metavar="BRANCH=JSON")
    parser.add_argument("--branch-stylus", nargs="*", default=[], metavar="BRANCH=STYLUS_ID")
    parser.add_argument("--require-auto-ready", action="store_true")
    return parser.parse_args()


def main():
    """
    主入口函数：加载、验证注册表并解析分支到 stylus 的映射。

    执行流程：
      1. 解析命令行参数。
      2. 加载并交叉验证注册表（load_registry + validate_registry）。
      3. 如果提供了 branches 或 branch-fit，解析分支到 stylus 的映射。
      4. 打印每个分支对应的 stylus_id 以及激活的标准球信息。

    Returns:
        成功时返回 0。
    """
    args = parse_args()
    registry = validate_registry(load_registry(args.registry))
    fit_values = parse_mapping(args.branch_fit, "--branch-fit")
    stylus_values = parse_mapping(args.branch_stylus, "--branch-stylus")
    branches = args.branches or sorted(fit_values)
    if branches:
        resolved = resolve_branch_styli(
            registry,
            branches,
            {branch: Path(path) for branch, path in fit_values.items()},
            stylus_values,
            args.reference_fit_json,
            args.require_auto_ready,
        )
        for branch in branches:
            print(f"{branch}: {resolved[branch]}")
    sphere_id = registry["active_standard_sphere_id"]
    sphere = registry["standard_spheres"][sphere_id]
    print(f"registry OK: sphere={sphere_id} center={sphere['sphere_center_mm']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
