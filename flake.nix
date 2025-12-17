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
        {
          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.typst
              pkgs.pnpm_10
              pkgs.nodejs_25
              pkgs.python314
              pkgs.uv
              pkgs.texlab
              (pkgs.texlive.combine {
                inherit (pkgs.texlive) scheme-medium todonotes;
              })
              pkgs.watchexec
            ];
          };
        };
    };
}
