---
license: apache-2.0
base_model: MemTensor/MemT-1.0-Nano
base_model_relation: quantized
quantized_by: UnstableLlama
tags:
- exl3
- memt
---

<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
  .dashboard-container { font-family: 'Inter', sans-serif; width: min(1500px, calc(100vw - 32px)); max-width: 100%; margin: 0 auto; box-sizing: border-box; background-color: #1a1b1e; background-image: radial-gradient(#2d2f34 1px, transparent 1px); background-size: 20px 20px; color: #e0e0e0; padding: 40px 24px; border: 1px solid #4a4d53; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
  .dashboard-header { margin-bottom: 35px; }
  .dashboard-header h1 { font-family: 'JetBrains Mono', monospace; color: #ffffff; font-size: 1.6em; margin: 0; padding-left: 15px; border-left: 4px solid #4dabf7; }
  .meta-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.8em; color: #4dabf7; background: rgba(77, 171, 247, 0.1); padding: 4px 12px; border: 1px solid rgba(77, 171, 247, 0.3); border-radius: 4px; margin-top: 12px; display: inline-block; }
  .content-panel { background-color: #25262b; border: 1px solid #4a4d53; border-radius: 8px; margin-bottom: 25px; overflow: hidden; }
  .panel-title { background-color: #4dabf7; color: #000000; padding: 8px 20px; font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }
  .panel-body { padding: 20px; }
  .repo-data-panel { padding: 14px 10px; }
  .repo-data-body { display: flex; flex-direction: column; align-items: center; gap: 20px; width: 100%; --edge-gap: 8px; }
  .repo-graph { display: block; width: min(1440px, calc(100% - (var(--edge-gap) * 2))); height: auto; margin: 0 auto; }
  .repo-graph + .repo-graph { margin-top: 1.25rem; }
  .table-wrapper { display: inline-block; margin: 0 auto; border: 1px solid #666a73; border-radius: 4px; overflow: hidden; max-width: calc(100% - (var(--edge-gap) * 2)); }
  .data-table { border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; width: auto; margin: 0; }
  .data-table th { text-align: left; color: #ffffff; background-color: #2d2f34; padding: 9px 12px; border-bottom: 2px solid #666a73; border-right: 1px solid #4a4d53; }
  .data-table td { padding: 7px 12px; border-bottom: 1px solid #4a4d53; border-right: 1px solid #4a4d53; }
  .data-table tr td:last-child, .data-table tr th:last-child { border-right: none; }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table tr:hover td { background-color: rgba(77, 171, 247, 0.05); }
  .link-style { color: #4dabf7; text-decoration: none; }
  .link-style:hover { text-decoration: underline; color: #ffffff; }
  .terminal-box { background-color: #0c0d0e; border: 1px solid #4a4d53; border-radius: 6px; padding: 18px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; color: #cbd5e0; white-space: pre-wrap; }
</style>

<div class="dashboard-container">

  <div class="dashboard-header">
    <h1>MemTensor / MemT-1.0-Nano</h1>
    <div class="meta-tag">ASSEMBLED BY: UnstableLlama</div>
  </div>

  <div class="content-panel">
    <div class="panel-title">Information</div>
    <div class="panel-body">
      exl3 builds of the <b><a class="link-style" href="https://huggingface.co/MemTensor/MemT-1.0-Nano">MemT-1.0-Nano</a></b>
      &#9888;&#65039; <b>PERSISTENT MEMORY MODEL</b> &#9888;&#65039; that works with TabbyAPI.
      ExLlamaV3 fork <b><a class="link-style" href="https://github.com/UnstableLlama/exl3-qlora">exl3-qlora</a></b> required.
      <br/><br/>
      MemT-1.0-Nano is Qwen3.6-35B-A3B plus a trained memory block on its ten full-attention layers: the model files earlier
      conversation turns into a 40 MiB latent memory and answers from it instead of re-reading the whole chat.
      <br/><br/>
      Each revision is the stock backbone quant from
      <a class="link-style" href="https://huggingface.co/yeasah/Qwen3.6-35B-A3B-exl3">yeasah/Qwen3.6-35B-A3B-exl3</a> (byte for byte, credit yeasah) plus the MemT
      files: <code>memt_block.safetensors</code> (bf16, 0.84 GiB), the MemT configs, and the release chat template with a small
      addition for client-side memory control. No requant.
    </div>
  </div>

  <div class="content-panel">
    <div class="panel-title">Repo Data</div>
    <div class="panel-body repo-data-body repo-data-panel">
      <img class="repo-graph" src="qb_kld.png" alt="Mean KL divergence vs bits per weight">
      <img class="repo-graph" src="qb_ppl.png" alt="Perplexity vs bits per weight">
      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>REVISION</th>
              <th>GiB</th>
              <th>KL DIV</th>
              <th>PPL</th>
              <th>BACKBONE QUANT</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><a class="link-style" href="https://huggingface.co/UnstableLlama/MemT-1.0-Nano-exl3/tree/3.00bpw">3.00bpw</a></td>
              <td>15.67</td>
              <td>0.0215</td>
              <td>1.3389</td>
              <td><a class="link-style" href="https://huggingface.co/yeasah/Qwen3.6-35B-A3B-exl3/tree/3.00bpw-H5">3.00bpw-H5</a></td>
            </tr>
            <tr>
              <td><a class="link-style" href="https://huggingface.co/UnstableLlama/MemT-1.0-Nano-exl3/tree/4.00bpw">4.00bpw</a></td>
              <td>19.64</td>
              <td>0.0069</td>
              <td>1.3207</td>
              <td><a class="link-style" href="https://huggingface.co/yeasah/Qwen3.6-35B-A3B-exl3/tree/4.00bpw">4.00bpw</a></td>
            </tr>
            <tr>
              <td><a class="link-style" href="https://huggingface.co/UnstableLlama/MemT-1.0-Nano-exl3/tree/5.00bpw">5.00bpw</a></td>
              <td>23.55</td>
              <td>0.0030</td>
              <td>1.3150</td>
              <td><a class="link-style" href="https://huggingface.co/yeasah/Qwen3.6-35B-A3B-exl3/tree/5.00bpw">5.00bpw</a></td>
            </tr>
            <tr>
              <td><a class="link-style" href="https://huggingface.co/UnstableLlama/MemT-1.0-Nano-exl3/tree/6.00bpw">6.00bpw</a></td>
              <td>27.47</td>
              <td>0.0018</td>
              <td>1.3116</td>
              <td><a class="link-style" href="https://huggingface.co/yeasah/Qwen3.6-35B-A3B-exl3/tree/6.00bpw">6.00bpw</a></td>
            </tr>
            <tr>
              <td><a class="link-style" href="https://huggingface.co/MemTensor/MemT-1.0-Nano">bf16</a></td>
              <td>63.61</td>
              <td>0.0000</td>
              <td>1.3126</td>
              <td>-</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="content-panel">
    <div class="panel-title">Usage</div>
    <div class="panel-body">
      Install the fork into TabbyAPI's venv and launch Tabby on a revision. Memory is on by default for this model.
      <div class="terminal-box">EXL3_MEMT_SNAPSHOT_DIR=./memt_snapshots ./venv/bin/python main.py --model-name ./MemT-1.0-Nano-exl3-3.00bpw \
  --draft-mode disabled --max-seq-len 32768 --cache-size 32768 --gpu-split 20 23 \
  --tool-format auto --reasoning-start-token "&lt;think&gt;" --reasoning-end-token "&lt;/think&gt;"</div>
      <br/>
      Chat normally. Type <code>/commit</code> to memorize the conversation so far (the prompt the model reads shrinks to what came
      after), or <code>/commit every 3</code> to let it memorize on its own. Edits roll memory back, regenerations don't memorize
      twice, chats don't share memory. Every setting, the request-level controls and the test scripts are in the fork's
      <a class="link-style" href="https://github.com/UnstableLlama/exl3-qlora/blob/master/doc/MEMT_README.md">doc/MEMT_README.md</a>.
      <br/><br/>
      Not yet measured: whether memorizing replies as the model generated them (reasoning included, the original
      MemT runtime's way, <code>EXL3_MEMT_COMMIT_GENERATED=1</code>) recalls better than memorizing the client's
      re-sent copy (the default). Both are supported; nobody has compared them.
    </div>
  </div>

  <div class="content-panel">
    <div class="panel-title">CLI Download</div>
    <div class="panel-body">
      <div class="terminal-box">hf download UnstableLlama/MemT-1.0-Nano-exl3 --revision "3.00bpw" --local-dir ./MemT-1.0-Nano-exl3-3.00bpw</div>
    </div>
  </div>

</div>
