{

  description = "A reproducible environment for learning certifiable controllers";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/ff1ea3a36c1dfafdf0490e0d144c5a0ed0891bb9";
  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
          jupyter = builtins.fetchGit {                         
              url = https://github.com/tweag/jupyterWith;       
              rev = "37cd8caefd951eaee65d9142544aa4bd9dfac54f";
            };

          overlays = [
            (import "${jupyter}/nix/python-overlay.nix")
            (import "${jupyter}/nix/overlay.nix")
          ];

          pkgs = import (nixpkgs) {
              config = {allowUnfree = true;};
              system = "x86_64-linux";
              inherit overlays;
            };
          extensions = (with pkgs.vscode-extensions; [
            ms-python.python
            ms-toolsai.jupyter
          ]);

          packages = ps: with ps; 
            [ numpy tqdm matplotlib scipy gym tensorflow mypy ipywidgets];

          pythonWithPackages = pkgs.python3.withPackages packages;

          iPython = pkgs.jupyterWith.kernels.iPythonWith {
            name = "test";
            python3 = pythonWithPackages;
            packages = packages;
          };

          jupyterlab = pkgs.jupyterWith.jupyterlabWith {
            kernels = [ iPython ];
          };
        
          vscodium-with-extensions = pkgs.vscode-with-extensions.override {
            vscode = pkgs.vscodium;
            vscodeExtensions = extensions;
          };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            pythonWithPackages
            jupyterlab
            vscodium-with-extensions
          ];
          # QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
        };
      }
    );
}
