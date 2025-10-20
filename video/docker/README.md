# 视频矩阵系统 - go2rtc

基于 go2rtc 的 RTSP 视频流解决方案，支持 HLS、MP4 和 WebRTC 格式。

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面       │    │   Nginx 代理     │    │   go2rtc 服务    │
│   (HTML/JS)     │◄──►│   (8081端口)     │◄──►│   (1984端口)     │
│                 │    │                 │    │                 │
│ • 视频矩阵展示   │    │ • 反向代理       │    │ • RTSP 接收     │
│ • 控制面板       │    │ • 负载均衡       │    │ • 流转换         │
│ • 状态监控       │    │ • 静态文件服务   │    │ • WebRTC 服务器  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │   RTSP 摄像头    │
                                              │   (4个通道)      │
                                              └─────────────────┘
```

## 快速开始

### 1. 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 2. 访问系统

- **主界面**: http://localhost:8081
- **go2rtc 原始界面**: http://localhost:8081/
- **API 状态**: http://localhost:8081/api/streams

### 3. 使用说明

1. 打开 http://localhost:8081
2. 点击"测试连接"验证服务器状态
3. 选择视频格式（推荐 HLS）
4. 点击"加载全部"或单独加载每个摄像头
5. 使用视频控制按钮进行操作

## 配置说明

### Docker Compose

```yaml
services:
  go2rtc:
    image: alexxit/go2rtc:latest
    ports:
      - "1984:1984"
    volumes:
      - ./config/go2rtc.yaml:/config/go2rtc.yaml
    networks:
      - video-network

  nginx:
    image: nginx:alpine
    ports:
      - "8081:8081"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./html:/usr/share/nginx/html:ro
    depends_on:
      - go2rtc
    networks:
      - video-network
```

### go2rtc 配置

```yaml
# config/go2rtc.yaml
api:
  listen: ":1984"
  username: ""
  password: ""

webrtc:
  listen: ":1984"
  ice_servers:
    - urls: ["stun:stun.l.google.com:19302"]

streams:
  camera1:
    - "rtsp://admin:password@192.168.1.100"
  camera2:
    - "rtsp://admin:password@192.168.1.101"
  camera3:
    - "rtsp://admin:password@192.168.1.102"
  camera4:
    - "rtsp://admin:password@192.168.1.103"

log:
  level: info
```

### Nginx 配置

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # 上游 go2rtc 服务
    upstream go2rtc {
        server go2rtc:1984;
    }

    server {
        listen 8081;
        server_name localhost;

        # 静态文件服务
        location / {
            root   /usr/share/nginx/html;
            index  index.html index.htm;
        }

        # API 代理
        location /api {
            proxy_pass http://go2rtc;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        # WebSocket 支持
        location /ws {
            proxy_pass http://go2rtc;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "Upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
```

## API 端点

### 流状态
- `GET /api/streams` - 获取所有流状态

### 视频流端点
- `GET /api/stream.m3u8?src={streamName}` - HLS 格式
- `GET /api/stream.mp4?src={streamName}` - MP4 格式
- `POST /api/webrtc?src={streamName}` - WebRTC 格式

### WebSocket
- `WS /ws` - WebSocket 连接

## 功能特性

### 前端界面
- 🎥 4 通道视频矩阵显示
- 🎛️ 完整的控制面板
- 📊 实时状态监控
- 📝 详细的日志系统
- 🎨 现代化响应式设计
- ⌨️ 键盘快捷键支持

### 视频格式支持
- ✅ **HLS (推荐)**: 低延迟，兼容性好
- ✅ **MP4**: 标准格式，支持广泛
- ✅ **WebRTC**: 超低延迟，实时性好

### 控制功能
- ▶️ 播放/暂停
- 🔇 静音/取消静音
- ⏹️ 停止视频
- 🔄 重新加载
- 📱 响应式布局

## 故障排除

### 常见问题

1. **容器启动失败**
   ```bash
   # 检查端口占用
   netstat -tuln | grep 8081
   netstat -tuln | grep 1984

   # 查看详细日志
   docker-compose logs go2rtc
   docker-compose logs nginx
   ```

2. **视频无法播放**
   - 检查 RTSP 源地址是否正确
   - 验证网络连接
   - 查看浏览器控制台错误
   - 测试不同的视频格式

3. **连接超时**
   - 确认 go2rtc 服务运行正常
   - 检查防火墙设置
   - 验证 nginx 代理配置

### 调试命令

```bash
# 测试 go2rtc 直接连接
curl http://localhost:1984/api/streams

# 测试 nginx 代理连接
curl http://localhost:8081/api/streams

# 测试 HLS 流
curl -I http://localhost:8081/api/stream.m3u8?src=camera1

# 测试 MP4 流
curl -I http://localhost:8081/api/stream.mp4?src=camera1

# 查看容器资源使用
docker stats

# 重启服务
docker-compose restart
```

## 性能优化

### go2rtc 优化
```yaml
# 在 go2rtc.yaml 中添加
streams:
  camera1:
    - "rtsp://admin:password@192.168.1.100#transport=tcp"
  camera2:
    - "rtsp://admin:password@192.168.1.101#transport=tcp"

webrtc:
  listen: ":1984"
  ice_servers:
    - urls: ["stun:stun.l.google.com:19302"]
  # 添加 TURN 服务器（如果需要）
  # - urls: ["turn:your-turn-server:3478"]
  #   username: "user"
  #   credential: "pass"
```

### Nginx 优化
```nginx
# 添加到 nginx.conf
http {
    # 缓存配置
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m inactive=60m;

    # 客户端配置
    client_max_body_size 100M;
    client_body_timeout 30s;

    # 代理配置
    proxy_connect_timeout 30s;
    proxy_send_timeout 30s;
    proxy_read_timeout 30s;

    # Gzip 压缩
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;
}
```

## 安全建议

1. **更改默认密码**
   - 为 RTSP 流设置强密码
   - 考虑启用 go2rtc 认证

2. **网络安全**
   - 使用防火墙限制访问
   - 考虑启用 HTTPS
   - 定期更新镜像

3. **访问控制**
   - 使用网络隔离
   - 监控访问日志
   - 限制 API 访问

## 扩展功能

### 添加更多摄像头
在 `config/go2rtc.yaml` 中添加新的流配置：

```yaml
streams:
  camera5:
    - "rtsp://admin:password@192.168.1.104"
  camera6:
    - "rtsp://admin:password@192.168.1.105"
```

### 自定义前端
修改 `html/index.html` 文件：
- 添加更多视频单元
- 自定义样式和布局
- 集成其他功能

### 集成其他系统
- 通过 API 集成到现有系统
- 使用 WebSocket 进行实时通信
- 添加录像和回放功能

## 技术栈

- **后端**: go2rtc, Nginx
- **前端**: HTML5, JavaScript, HLS.js
- **容器**: Docker, Docker Compose
- **视频**: RTSP, HLS, MP4, WebRTC
- **网络**: HTTP/HTTPS, WebSocket

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请提交 Issue 或联系维护者。