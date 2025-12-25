## jiarui 的 V2Ray / Xray 配置说明

- **根目录 (`$jiarui`)**: `/cephfs/shared/jiarui`
- **本项目路径**: `/cephfs/shared/jiarui/v2ray`
- **核心文件**: 
  - V2Ray: `/cephfs/shared/jiarui/v2ray/bin/v2ray`
  - Xray: `/cephfs/shared/jiarui/v2ray/bin/xray`
- **配置文件**: 
  - V2Ray: `/cephfs/shared/jiarui/v2ray/config.json` (使用 tcp/trojan 协议)
  - Xray: `/cephfs/shared/jiarui/v2ray/config2.json` (使用 xhttp/vless 协议)

### 一、启动代理服务（建议在 tmux 中后台运行）

#### 方式 1：使用 V2Ray（标准协议）

1. 创建名为 `proxy` 的 tmux 会话并在后台跑 V2Ray：

```bash
tmux new-session -d -s proxy "cd /cephfs/shared/jiarui/v2ray && ./bin/v2ray run -c ./config.json"
```

2. 之后如需查看日志或操作：

```bash
tmux attach -t proxy   # 进入会话
tmux detach            # 按 Ctrl+b d 退出回到后台
```

#### 方式 2：使用 Xray（支持 xhttp 等新协议）

**注意**: 如果服务器使用 xhttp、XTLS 等 Xray 特有协议，必须使用 Xray 而非 V2Ray。

1. 创建名为 `proxy2` 的 tmux 会话运行 Xray：

```bash
tmux new-session -d -s proxy2 "cd /cephfs/shared/jiarui/v2ray && ./xray/xray run -c ./config2.json"
```

2. 查看运行状态：

```bash
tmux attach -t proxy2   # 进入会话查看日志
# 按 Ctrl+b d 退出回到后台
```

3. 停止 Xray：

```bash
tmux kill-session -t proxy2
```

### 二、设置 / 取消终端代理

- **开启代理（当前 shell 生效）**：

```bash
source /cephfs/shared/jiarui/v2ray/network_setup.sh
```

- **关闭代理（清空相关环境变量）**：

```bash
source /cephfs/shared/jiarui/v2ray/unset_network.sh
```

当前配置：
- HTTP/HTTPS 代理：`http://127.0.0.1:10810`
- SOCKS5 代理：`socks5://127.0.0.1:10809`

**注意**：无论使用 V2Ray 还是 Xray，代理端口都是一样的，所以使用同一套网络设置脚本即可。

### 三、V2Ray vs Xray 说明

| 特性 | V2Ray (config.json) | Xray (config2.json) |
|------|---------------------|---------------------|
| **传输协议** | tcp, ws, h2 等标准协议 | 支持 xhttp, XTLS 等新协议 |
| **适用场景** | 使用传统 trojan/vmess 节点 | 使用新型 vless+xhttp+reality 节点 |
| **端口配置** | 10809(SOCKS), 10810(HTTP) | 10809(SOCKS), 10810(HTTP) |
| **兼容性** | 标准协议，兼容性好 | Xray 特有协议，需服务端支持 |

**如何选择**：
- 如果遇到 `unknown transport protocol: xhttp` 错误 → 使用 Xray
- 如果服务器配置中有 XTLS、xhttp、Reality 等关键词 → 使用 Xray
- 传统节点（trojan/vmess over tcp/ws） → 使用 V2Ray 即可


