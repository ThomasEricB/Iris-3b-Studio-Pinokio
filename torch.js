// Iris-3B needs torch >= 2.7.1 (the stock torch.js pins 2.7.0). 2.8.0 also covers RTX 50-series (cu128).
module.exports = {
  run: [
    // nvidia (windows + linux)
    {
      "when": "{{gpu === 'nvidia' && platform !== 'darwin'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": "uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128"
      },
      "next": null
    },
    // amd linux (rocm)
    {
      "when": "{{gpu === 'amd' && platform === 'linux'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": "uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/rocm6.4"
      },
      "next": null
    },
    // apple silicon mac (MPS)
    {
      "when": "{{platform === 'darwin' && arch === 'arm64'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": "uv pip install torch==2.8.0 torchvision==0.23.0"
      },
      "next": null
    },
    // everything else (amd windows, intel/no gpu): cpu
    {
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": "uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu"
      }
    }
  ]
}
