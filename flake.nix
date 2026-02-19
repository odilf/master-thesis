{
  description = "Description for the project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        { pkgs, ... }:
        let
          code = [
            pkgs.pnpm_10
            pkgs.nodejs_25
            pkgs.python314
            pkgs.uv
            pkgs.watchexec
          ];

          tex = [
            pkgs.just
            pkgs.watchexec
            pkgs.texlab
            (pkgs.texlive.combine {
              inherit (pkgs.texlive)
                scheme-small
                biblatex
                geometry
                libertinus
                libertinust1math
                todonotes
                ;
            })
            pkgs.watchexec
          ];
        in
        {
          devShells = {
            code = pkgs.mkShell { packages = code; };
            tex = pkgs.mkShell { packages = tex; };
            default = pkgs.mkShell { packages = code ++ tex; };
          };
        };
    };
}
