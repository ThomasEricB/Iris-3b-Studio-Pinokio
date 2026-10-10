module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Iris-3B needs PyTorch >= 2.7.1, which has no Intel-Mac build
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "Intel Macs are not supported: Iris-3B needs PyTorch 2.7.1+, which has no Intel macOS build."
      },
      next: null
    },
    // Official Iris-3B code, used unmodified
    {
      when: "{{!exists('app/iris-3b')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/speridlabs/iris-3b.git app/iris-3b"
      }
    },
    // Iris-3B requires Python >= 3.11; uv fetches a managed 3.11 on every OS
    {
      when: "{{!exists('app/env')}}",
      method: "shell.run",
      params: {
        path: "app",
        message: "uv venv env --python 3.11 --seed"
      }
    },
    // PyTorch first, so the requirements below never pull a CPU-only build from PyPI
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install --no-deps -e iris-3b"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Installed! Each tab downloads its weights on first use (~12 GB each, plus ~9 GB text encoder for Text to Image). Click Start."
      }
    }
  ]
}
