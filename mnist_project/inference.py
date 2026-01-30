import os
import argparse
import torch
import torchvision.utils as vutils


def load_generator_torchscript(model_path, device="cpu"):
    device = torch.device(device)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def generate_images(generator, digit, outdir, n_samples=10, z_dim=100, device="cpu"):
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(device)

    with torch.no_grad():
        for i in range(n_samples):
            noise = torch.randn(1, z_dim, device=device)
            label = torch.tensor([digit], device=device)

            image = generator(noise, label)

            # [-1,1] → [0,1]
            image = (image + 1) / 2

            save_path = os.path.join(outdir, f"digit_{digit}_sample_{i}.png")
            vutils.save_image(image, save_path)

    print(f"{n_samples} imagens do digito {digit} salvas em {outdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Caminho do generator (.pt)")
    parser.add_argument("--digit", type=int, required=True, help="Digito a ser gerado (0-9)")
    parser.add_argument("--outdir", required=True, help="Diretório de saida")
    parser.add_argument("--samples", type=int, default=10, help="Numero de imagens a gerar")
    parser.add_argument("--device", default="cpu", help="cpu ou cuda")

    args = parser.parse_args()

    G = load_generator_torchscript(args.model, args.device)
    generate_images(G, args.digit, args.outdir, args.samples, device=args.device)


if __name__ == "__main__":
    main()
