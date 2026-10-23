#include <dobot_bringup/command.h>
#include <iostream>
#include <chrono>
#include <thread>

namespace
{
constexpr auto kDashboardReplyTimeout = std::chrono::milliseconds(3000);
std::mutex g_dashboard_command_mutex;

std::string receiveDashboardReply(std::shared_ptr<TcpClient> &tcp)
{
    std::string reply;
    const auto deadline = std::chrono::steady_clock::now() + kDashboardReplyTimeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        char chunk[256] = {};
        uint32_t has_read = 0;
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        const auto timeout_ms = static_cast<uint32_t>(
            std::max<int64_t>(1, std::min<int64_t>(100, remaining.count())));
        tcp->tcpRecv(chunk, sizeof(chunk), has_read, timeout_ms);
        if (has_read > 0)
        {
            reply.append(chunk, has_read);
            const auto terminator = reply.find(';');
            if (terminator != std::string::npos)
            {
                reply.resize(terminator + 1);
                return reply;
            }
        }
    }
    throw TcpClientException("dashboard reply timeout after 3000 ms");
}

int32_t parseDashboardErrorId(const std::string &reply)
{
    const auto comma = reply.find(',');
    if (comma == std::string::npos)
        throw TcpClientException("invalid dashboard reply: missing error-id delimiter: " + reply);

    const std::string token = reply.substr(0, comma);
    size_t consumed = 0;
    int32_t error_id = 0;
    try
    {
        error_id = std::stoi(token, &consumed);
    }
    catch (const std::exception &)
    {
        throw TcpClientException("invalid dashboard error id in reply: " + reply);
    }
    if (consumed != token.size())
        throw TcpClientException("invalid dashboard error id in reply: " + reply);
    return error_id;
}
} // namespace

CRCommanderRos2::CRCommanderRos2(const std::string &ip)
    : current_joint_{}, tool_vector_{}, is_running_(false)
{
    is_running_ = false;
    real_time_data_ = std::make_shared<RealTimeData>();
    real_time_tcp_ = std::make_shared<TcpClient>(ip, 30004);
    dash_board_tcp_ = std::make_shared<TcpClient>(ip, 29999);
}

CRCommanderRos2::~CRCommanderRos2()
{
    is_running_ = false;
    thread_->join();
}

void CRCommanderRos2::getCurrentJointStatus(double *joint)
{
    mutex_.lock();
    memcpy(joint, current_joint_, sizeof(current_joint_));
    mutex_.unlock();
}

void CRCommanderRos2::getToolVectorActual(double *val)
{
    mutex_.lock();
    memcpy(val, tool_vector_, sizeof(tool_vector_));
    mutex_.unlock();
}

void CRCommanderRos2::recvTask()
{
    uint32_t has_read;
    while (is_running_)
    {
        if (real_time_tcp_->isConnect())
        {
            try
            {
                uint8_t *tmpData = reinterpret_cast<uint8_t *>(real_time_data_.get());
                if (real_time_tcp_->tcpRecv(tmpData, sizeof(RealTimeData), has_read, 5000))
                {

                    if (real_time_data_->len != 1440)
                        continue;

                    mutex_.lock();
                    for (uint32_t i = 0; i < 6; i++)
                        current_joint_[i] = deg2Rad(real_time_data_->q_actual[i]);

                    memcpy(tool_vector_, real_time_data_->tool_vector_actual, sizeof(tool_vector_));
                    mutex_.unlock();
                }
                else
                {
                    std::cout << "tcp recv timeout" << std::endl;
                }
            }
            catch (const TcpClientException &err)
            {
                real_time_tcp_->disConnect();
                std::cout << "tcp recv error :" << std::endl;
            }
        }
        else
        {
            try
            {
                real_time_tcp_->connect();
            }
            catch (const TcpClientException &err)
            {
                std::cout << "tcp recv Error : %s" << std::endl;
                sleep(3);
            }
        }

        if (!dash_board_tcp_->isConnect())
        {
            try
            {
                dash_board_tcp_->connect();
            }
            catch (const TcpClientException &err)
            {

                std::cout << "tcp recv ERROR : %s" << std::endl;
                sleep(3);
            }
        }
    }
}

void CRCommanderRos2::init()
{
    try
    {
        is_running_ = true;
        thread_ = std::unique_ptr<std::thread>(new std::thread(&CRCommanderRos2::recvTask, this));
    }
    catch (const TcpClientException &err)
    {
        std::cout << "Commander : %s" << std::endl;
    }
}
void CRCommanderRos2::doTcpCmd(std::shared_ptr<TcpClient> &tcp, const char *cmd, int32_t &err_id,
                               std::vector<std::string> &result)
{
    err_id = -1;
    const auto current_time = std::chrono::system_clock::now();
    const auto current_time_ms =
        std::chrono::time_point_cast<std::chrono::milliseconds>(current_time);
    std::cout << "time: " << current_time_ms.time_since_epoch().count()
              << "  tcp send cmd :" << cmd << std::endl;

    const std::string framed_command = std::string(cmd) + "\n";
    tcp->tcpSend(framed_command.data(), framed_command.size());
    const std::string reply = receiveDashboardReply(tcp);
    err_id = parseDashboardErrorId(reply);
    result.clear();
    result.push_back(reply);
    std::cout << "ErrorID: " << err_id << std::endl;
    std::cout << "tcp recv feedback : " << reply << std::endl;
}


void CRCommanderRos2::doTcpCmd_f(std::shared_ptr<TcpClient> &tcp, const char *cmd, int32_t &err_id,std::string &mode_id,
                               std::vector<std::string> &result)
{
    err_id = -1;
    mode_id.clear();
    const auto current_time = std::chrono::system_clock::now();
    const auto current_time_ms =
        std::chrono::time_point_cast<std::chrono::milliseconds>(current_time);
    std::cout << "time: " << current_time_ms.time_since_epoch().count()
              << "  tcp send cmd :" << cmd << std::endl;

    const std::string framed_command = std::string(cmd) + "\n";
    tcp->tcpSend(framed_command.data(), framed_command.size());
    const std::string reply = receiveDashboardReply(tcp);
    err_id = parseDashboardErrorId(reply);
    const auto payload_start = reply.find('{');
    const auto payload_end = reply.find('}', payload_start);
    if (payload_start == std::string::npos || payload_end == std::string::npos)
        throw TcpClientException("invalid dashboard reply payload: " + reply);
    mode_id = reply.substr(payload_start, payload_end - payload_start + 1);
    result.clear();
    result.push_back(reply);
    std::cout << "ErrorID: " << err_id << std::endl;
    std::cout << "tcp recv feedback : " << reply << std::endl;
}

bool CRCommanderRos2::callRosService(const std::string cmd, int32_t &err_id)
{
    std::lock_guard<std::mutex> lock(g_dashboard_command_mutex);
    err_id = -1;
    try
    {
        std::vector<std::string> result_;
        doTcpCmd(this->dash_board_tcp_, cmd.c_str(), err_id, result_);
        return true;
    }
    catch (const TcpClientException &err)
    {
        std::cout << "dashboard command failed: " << err.what() << std::endl;
        return false;
    }
}
bool CRCommanderRos2::callRosService_f(const std::string cmd, int32_t &err_id,std::string &mode_id)
{
    std::lock_guard<std::mutex> lock(g_dashboard_command_mutex);
    err_id = -1;
    mode_id.clear();
    try
    {
        std::vector<std::string> result_;
        doTcpCmd_f(this->dash_board_tcp_, cmd.c_str(), err_id,mode_id, result_);
        return true;
    }
    catch (const TcpClientException &err)
    {
        std::cout << "dashboard command failed: " << err.what() << std::endl;
        return false;
    }
}
bool CRCommanderRos2::callRosService(const std::string cmd, int32_t &err_id, std::vector<std::string> &result_)
{
    std::lock_guard<std::mutex> lock(g_dashboard_command_mutex);
    err_id = -1;
    try
    {
        doTcpCmd(this->dash_board_tcp_, cmd.c_str(), err_id, result_);
        return true;
    }
    catch (const TcpClientException &err)
    {
        std::cout << "dashboard command failed: " << err.what() << std::endl;
        return false;
    }
}

bool CRCommanderRos2::isEnable() const
{
    return real_time_data_->robot_mode == 5;
}

bool CRCommanderRos2::isConnected() const
{
    return dash_board_tcp_->isConnect() && real_time_tcp_->isConnect();
}

uint16_t CRCommanderRos2::getRobotMode() const
{
    return real_time_data_->robot_mode;
}

std::shared_ptr<RealTimeData> CRCommanderRos2::getRealData() const
{
    return real_time_data_;
}
