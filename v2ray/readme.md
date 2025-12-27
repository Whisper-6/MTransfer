## jiarui 的 Xray 配置说明（仅 Xray）

* **根目录 (`$jiarui`)**：`/cephfs/shared/jiarui`
* **本项目路径**：`/cephfs/shared/jiarui/v2ray`
* **Xray 核心文件**：

  ```text
  /cephfs/shared/jiarui/v2ray/bin/xray
  ```
* **Xray 配置文件**（使用 xhttp / vless 等新协议）：

  ```text
  /cephfs/shared/jiarui/v2ray/config2.json
  ```

---

## 一、启动 Xray 代理服务（建议在 tmux 中后台运行）

> ⚠️ 服务器使用 **xhttp / XTLS / Reality / vless** 等协议，**必须使用 Xray**

### 1️⃣ 在 tmux 中启动 Xray

创建名为 `proxy2` 的 tmux 会话并后台运行：

```bash
tmux new-session -d -s proxy2 "cd /cephfs/shared/jiarui/v2ray && ./bin/xray run -c ./config2.json"
```

---

### 2️⃣ 查看运行状态 / 日志

```bash
tmux attach -t proxy2
```

* 查看日志是否正常
* 退出但保持后台运行：`Ctrl + b` → `d`

---

### 3️⃣ 停止 Xray

```bash
tmux kill-session -t proxy2
```

---

## 二、设置 / 取消终端代理（对当前 shell 生效）

### ✅ 开启代理

```bash
source /cephfs/shared/jiarui/v2ray/network_setup.sh
```

### ❌ 关闭代理（清空代理环境变量）

```bash
source /cephfs/shared/jiarui/v2ray/unset_network.sh
```

---

### 当前代理端口配置

* **SOCKS5**：`socks5://127.0.0.1:10809`
* **HTTP / HTTPS**：`http://127.0.0.1:10810`

> 端口由 `config2.json` 决定，网络脚本与 Xray 配置保持一致即可。

---

## 三、常见问题与判断

### ✔ 什么时候一定要用 Xray？

* 配置中出现以下关键词之一：

  * `xhttp`
  * `XTLS`
  * `Reality`
  * `vless`
* 报错类似：

  ```text
  unknown transport protocol: xhttp
  ```

➡️ **结论：只能用 Xray**

---

## 四、使用流程速查（TL;DR）

```bash
# 1. 启动 Xray
tmux new -d -s proxy2 "cd /cephfs/shared/jiarui/v2ray && ./bin/xray run -c ./config2.json"

# 2. 开启代理
source /cephfs/shared/jiarui/v2ray/network_setup.sh

# 3. 用完关闭
source /cephfs/shared/jiarui/v2ray/unset_network.sh
tmux kill-session -t proxy2
```
