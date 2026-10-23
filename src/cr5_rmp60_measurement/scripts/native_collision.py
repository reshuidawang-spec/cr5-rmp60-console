"""Local FCL bridge with conservative convex hulls for every robot mesh."""
from __future__ import annotations
import ctypes as C
import fcntl
import hashlib
from pathlib import Path
import subprocess
import tempfile
from ament_index_python.packages import get_package_prefix
import numpy as np
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1]

class NativeCollision:
    def __init__(self):
        source=ROOT/'cpp/sphere_collision_distance.cpp'
        self.source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        shapes_prefix=Path(get_package_prefix('geometric_shapes'))
        eigen_prefix=Path(get_package_prefix('eigen_stl_containers'))
        build_flags=[f'-I{shapes_prefix}/include/geometric_shapes',
                     f'-I{eigen_prefix}/include/eigen_stl_containers',
                     f'-L{shapes_prefix}/lib',f'-Wl,-rpath,{shapes_prefix}/lib','-lgeometric_shapes']
        build_hash=hashlib.sha256((self.source_hash+repr(build_flags)).encode()).hexdigest()
        cache=Path(tempfile.gettempdir())/'cr5_collision_native'
        cache.mkdir(mode=0o700,exist_ok=True)
        library=cache/f'collision_{build_hash[:20]}.so'
        with (cache/'build.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not library.exists():
                flags=subprocess.check_output(['pkg-config','--cflags','--libs','fcl'],text=True).split()
                output=library.with_suffix('.tmp.so')
                result=subprocess.run(['g++','-O2','-std=c++17','-fPIC','-shared',str(source),'-o',str(output),*build_flags,*flags],capture_output=True,text=True)
                if result.returncode:raise RuntimeError('collision checker build failed: '+result.stderr[-2000:])
                output.replace(library)
        self.lib=C.CDLL(str(library));self.handles=[]
        arr=C.POINTER(C.c_double)
        self.lib.cr5_shape.argtypes=[C.c_int,arr];self.lib.cr5_shape.restype=C.c_void_p
        self.lib.cr5_convex.argtypes=[arr,C.c_int,C.POINTER(C.c_int),C.c_int,C.c_int];self.lib.cr5_convex.restype=C.c_void_p
        self.lib.cr5_mesh_vertices.argtypes=[C.c_char_p,arr,C.c_int];self.lib.cr5_mesh_vertices.restype=C.c_int
        self.lib.cr5_distance.argtypes=[C.c_void_p,arr,C.c_void_p,arr];self.lib.cr5_distance.restype=C.c_double
        self.lib.cr5_free.argtypes=[C.c_void_p]

    def keep(self,handle):
        if not handle:raise ValueError('invalid native collision geometry')
        self.handles.append(handle);return handle

    def shape(self,kind,dimensions):
        dimensions=np.ascontiguousarray(dimensions,dtype=float)
        if not np.isfinite(dimensions).all() or np.any(dimensions<=0):raise ValueError('invalid collision dimensions')
        return self.keep(self.lib.cr5_shape(kind,dimensions.ctypes.data_as(C.POINTER(C.c_double))))

    def convex(self,vertices):
        points=np.unique(np.asarray(vertices,dtype=float),axis=0)
        if not np.isfinite(points).all():raise ValueError('nonfinite collision vertices')
        hull=ConvexHull(points)
        points=points[hull.vertices]
        hull=ConvexHull(points)
        points=np.ascontiguousarray(points,dtype=float)
        faces=[]
        # FCL requires counterclockwise outward-facing polygons.
        for triangle,equation in zip(hull.simplices,hull.equations):
            a,b,c=map(int,triangle)
            if np.cross(points[b]-points[a],points[c]-points[a])@equation[:3]<0:b,c=c,b
            faces.extend([3,a,b,c])
        indices=np.ascontiguousarray(faces,dtype=np.int32)
        return self.keep(self.lib.cr5_convex(points.ctypes.data_as(C.POINTER(C.c_double)),len(points),
            indices.ctypes.data_as(C.POINTER(C.c_int)),len(hull.simplices),len(indices)))

    def vertices(self,path):
        count=self.lib.cr5_mesh_vertices(str(path).encode(),None,0)
        if count<=0:raise ValueError(f'cannot load collision mesh: {path}')
        values=np.empty((count,3),dtype=float)
        if self.lib.cr5_mesh_vertices(str(path).encode(),values.ctypes.data_as(C.POINTER(C.c_double)),count)!=count:
            raise ValueError(f'cannot read collision mesh: {path}')
        return values

    def distance(self,a,ta,b,tb):
        ta=np.ascontiguousarray(ta,dtype=float);tb=np.ascontiguousarray(tb,dtype=float)
        d=self.lib.cr5_distance(a,ta.ctypes.data_as(C.POINTER(C.c_double)),b,tb.ctypes.data_as(C.POINTER(C.c_double)))
        if not np.isfinite(d):raise ValueError('collision distance failed')
        return float(d)

    def close(self):
        for handle in self.handles:self.lib.cr5_free(handle)
        self.handles.clear()

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'handles'):
            self.close()
