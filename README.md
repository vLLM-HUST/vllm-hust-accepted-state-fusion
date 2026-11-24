# Accepted Token State Fusion

已接受 Token 状态写回融合是独立登记的实验 MOD。它依赖冻结 R10 的 BetterScale
状态路径，在 H2D 后把已接受 token 计数的收集、活跃前缀写回和尾部填充融合成一次 Triton 调用。
独立仓库管理这项改动的代码、构建身份和评测记录。

## 试验状态

- 组件验证覆盖 144 组 CPU 一致性测试，以及 7 种昇腾 910B2 形状的即时执行和每种 3 次图重放。
  8／10 活跃行的同步完成时间由 162.89／158.30 μs 降至 69.22／67.50 μs。
- R10 整模十题同卡筛选中，输出吞吐下降 6.39%，P95 每 token 耗时增加 3.22%，答对从 10 题降到
  6 题。两次同卡原生对照分别答对 7 和 10 题。该版本未通过整模筛选，默认关闭。

[版本化证据](evidence/summary.json)

## 构建实验覆盖层

适用工况是 vLLM-HUST Engine R10 2026.11.21.10、Qwen3.5-35B-A3B BF16、
双 Ascend 910B2 TP2、E36/R54、FULL 和 MTP2。

```sh
python tools/build_overlay.py /path/to/admitted/r10/sdk ./build
python -m unittest discover -s tests -v
```

构建器会校验三份宿主输入和三份生成文件的完整哈希，另行写出覆盖清单。宿主 SDK
由使用者在有访问授权的环境中提供。构建器只生成文件，激活由隔离的实验启动器在冻结基线准入后处理。

按清单分别将生成文件绑定到私有挂载命名空间内的对应位置。状态写回和启动预编译使用同一 Triton
入口，两个 TP rank 都须在服务封存日志中出现 `accepted_state_publication: 1`，并在真实预热请求中
出现 `R10_RESEARCH_ACCEPTED_STATE_PUBLICATION`。对照使用新进程及原始 SDK。结束实验启动器即可解除
私有挂载。

## English

This separately versioned experimental MOD depends on the BetterScale resident-state path in the
admitted R10 SDK. The builder emits the exact state-publication, compile-only warmup and local
contract overlays used by the sealed experiment. The device component improved. The ten-case
end-to-end integration screen did not pass. The MOD is disabled by default.
