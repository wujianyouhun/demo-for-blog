# RTSP到Web流媒体转换器 - 安装指南

## 系统要求

- Windows/Linux/macOS
- Python 3.7+
- FFmpeg（必须安装）
- 网络摄像头或RTSP流媒体源

## 安装步骤

### 1. 安装FFmpeg

#### Windows
1. 下载FFmpeg：https://ffmpeg.org/download.html
2. 解压到 `C:\ffmpeg` 或其他目录
3. 将FFmpeg的 `bin` 目录添加到系统PATH环境变量
4. 验证安装：在命令行运行 `ffmpeg -version`

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

### 2. 安装Python依赖

```bash
pip install websockets
```

或者安装完整的依赖：
```bash
pip install -r requirements.txt
```

### 3. 配置RTSP源

编辑 `ffmpeg_config.json` 文件，修改RTSP地址：

```json
{
  "rtsp_config": {
    "url": "rtsp://username:password@camera_ip:port/stream"
  },
  "default_stream_config": {
    "video_codec": "libx264",
    "audio_codec": "aac",
    "bitrate": "2000k",
    "resolution": "1280x720",
    "framerate": 25,
    "preset": "ultrafast",
    "tune": "zerolatency",
    "output_formats": ["hls", "dash"]
  }
}
```

### 4. 测试系统

运行系统测试：
```bash
python test_system.py
```

如果测试失败，尝试自动修复：
```bash
python test_system.py --fix
```

## 使用方法

### 启动服务

#### 方法1：使用启动脚本（推荐）
```bash
# Windows
start_streaming.bat

# Linux/macOS
./start_streaming.sh
```

#### 方法2：直接运行Python脚本
```bash
# 基础版本
python rtsp_to_web.py

# 高级版本（推荐）
python web_api_server.py
```

### 访问Web界面

打开浏览器访问：http://localhost:8080

### API接口

#### 获取所有流
```bash
curl http://localhost:8080/api/streams
```

#### 创建新流
```bash
curl -X POST http://localhost:8080/api/streams \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "camera1",
    "rtsp_url": "rtsp://admin:password@192.168.1.100/stream",
    "config": {
      "resolution": "1280x720",
      "bitrate": "2000k"
    }
  }'
```

#### 启动流
```bash
curl -X POST http://localhost:8080/api/streams/camera1/start
```

#### 停止流
```bash
curl -X POST http://localhost:8080/api/streams/camera1/stop
```

#### 获取系统统计
```bash
curl http://localhost:8080/api/stats
```

## 配置选项

### 视频质量设置

```json
{
  "stream_settings": {
    "video_codec": "libx264",
    "audio_codec": "aac",
    "bitrate": "2000k",
    "resolution": "1280x720",
    "framerate": 25,
    "preset": "ultrafast",
    "tune": "zerolatency"
  }
}
```

### 输出格式设置

```json
{
  "output_formats": {
    "hls": {
      "enabled": true,
      "segment_time": 4,
      "segment_list_size": 6
    },
    "dash": {
      "enabled": true,
      "segment_duration": 4
    }
  }
}
```

## 故障排除

### 常见问题

1. **FFmpeg未找到**
   - 确保FFmpeg已正确安装
   - 检查PATH环境变量
   - 在配置文件中指定FFmpeg完整路径

2. **RTSP连接失败**
   - 检查RTSP URL是否正确
   - 确认网络连接
   - 验证摄像头凭据
   - 检查防火墙设置

3. **端口被占用**
   - 修改配置文件中的端口号
   - 检查是否有其他程序占用端口

4. **播放延迟高**
   - 调整 `segment_time` 参数
   - 使用 `ultrafast` 预设
   - 降低视频比特率

### 性能优化

#### 低延迟设置
```json
{
  "stream_settings": {
    "preset": "ultrafast",
    "tune": "zerolatency",
    "segment_time": 2
  }
}
```

#### 高质量设置
```json
{
  "stream_settings": {
    "bitrate": "4000k",
    "resolution": "1920x1080",
    "preset": "fast"
  }
}
```

## 安全建议

1. 使用强密码保护RTSP流
2. 不要在配置文件中暴露敏感信息
3. 定期更新FFmpeg版本
4. 在生产环境中添加身份验证

## 支持的浏览器

- Chrome 65+
- Firefox 60+
- Safari 12+
- Edge 79+

## 技术支持

如有问题，请检查：
1. 日志文件：`stream_manager.log`
2. 控制台输出
3. 浏览器开发者工具网络面板

## 许可证

本项目基于MIT许可证开源。