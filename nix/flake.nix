{
  description = "Jupyter dev environment flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs, nixpkgs-unstable }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    pkgs-unstable = nixpkgs-unstable.legacyPackages.${system};

    python = pkgs.python314;
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python
        uv

        pandoc
        tinymist
        typst
        typstyle
        typst-live
      ];

      shellHook = ''
        export UV_PYTHON_PREFERENCE="only-system";
        export UV_PYTHON=${python}
      '';

      env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.libz
      ];

    };
  };
}

