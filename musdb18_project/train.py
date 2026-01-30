import torch
import musdb
import numpy as np
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class UNetGenerator(nn.Module):
    
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        
        # Encoder com dropout para regularizacao
        self.enc1 = self._conv_block(in_channels, 64, normalize=False, dropout=False)
        self.enc2 = self._conv_block(64, 128, dropout=True)
        self.enc3 = self._conv_block(128, 256, dropout=True)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512, dropout=True)
        
        # Decoder
        self.dec3 = self._upconv_block(512 + 256, 256, dropout=True)
        self.dec2 = self._upconv_block(256 + 128, 128, dropout=True)
        self.dec1 = self._upconv_block(128 + 64, 64, dropout=False)
        
        # Camada final
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()  # Melhor para espectrogramas
        )
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch, out_ch, normalize=True, dropout=False):
        """Bloco convolucional"""
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if normalize:
            layers.insert(1, nn.InstanceNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout2d(0.3))  # Regularizacao
        return nn.Sequential(*layers)
    
    def _upconv_block(self, in_ch, out_ch, dropout=False):
        """Bloco de upsampling"""
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.3))
        return nn.Sequential(*layers)
    
    def _match_dimensions(self, x, target):
        """Ajusta dimensões"""
        if x.shape[2:] != target.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=target.shape[2:], mode='bilinear', align_corners=False
            )
        return x
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder
        d3 = self.dec3(torch.cat([self._match_dimensions(b, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match_dimensions(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match_dimensions(d2, e1), e1], dim=1))
        
        out = self.final(d1)
        
        if out.shape[2:] != x.shape[2:]:
            out = self._match_dimensions(out, x)
        
        return out


class Discriminator(nn.Module):
   
    def __init__(self, in_channels=8):
        super().__init__()
        
        # Spectral Normalization para estabilidade
        self.model = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # Sem sigmoid - usa BCEWithLogitsLoss
        )
    
    def forward(self, mixture, target):
        if mixture.shape != target.shape:
            target = torch.nn.functional.interpolate(
                target, size=mixture.shape[2:], mode='bilinear', align_corners=False
            )
        x = torch.cat([mixture, target], dim=1)
        return self.model(x)


class MUSDB18Dataset(Dataset):
   
    def __init__(self, musdb_root, subset='train', target_stem='vocals', 
                 n_fft=2048, hop_length=512, duration=6.0, sample_rate=44100,
                 use_wav=False):
        self.mus = musdb.DB(root=musdb_root, subsets=subset, is_wav=use_wav, download=True)
        self.target_stem = target_stem
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.sample_rate = sample_rate
        self.chunk_size = int(duration * sample_rate)
        
    def __len__(self):
        return len(self.mus) * 10
    
    def __getitem__(self, idx):
        track_idx = idx % len(self.mus)
        track = self.mus[track_idx]
        
        mixture = track.audio.T
        target = track.targets[self.target_stem].audio.T
        
        # Extrai chunk
        max_start = max(0, mixture.shape[1] - self.chunk_size)
        if max_start > 0:
            start = np.random.randint(0, max_start)
        else:
            start = 0
        
        mixture_chunk = mixture[:, start:start + self.chunk_size]
        target_chunk = target[:, start:start + self.chunk_size]
        
        # Padding
        if mixture_chunk.shape[1] < self.chunk_size:
            pad = self.chunk_size - mixture_chunk.shape[1]
            mixture_chunk = np.pad(mixture_chunk, ((0,0), (0,pad)))
            target_chunk = np.pad(target_chunk, ((0,0), (0,pad)))
        
        # Normaliza áudio antes de converter
        mixture_chunk = mixture_chunk / (np.abs(mixture_chunk).max() + 1e-8)
        target_chunk = target_chunk / (np.abs(target_chunk).max() + 1e-8)
        
        # Converte para espectrograma
        mixture_spec = self._to_spectrogram(mixture_chunk)
        target_spec = self._to_spectrogram(target_chunk)
        
        return mixture_spec, target_spec
    
    def _to_spectrogram(self, audio):
        """Converte para espectrograma normalizado"""
        audio_tensor = torch.from_numpy(audio).float()
        specs = []
        
        for ch in range(audio_tensor.shape[0]):
            spec = torch.stft(
                audio_tensor[ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft),
                return_complex=True
            )
            
            mag = torch.abs(spec)
            phase = torch.angle(spec)
            
            # Normaliza magnitude para [0, 1]
            mag = mag / (mag.max() + 1e-8)

            # Normaliza fase para [0, 1]
            phase = (phase + np.pi) / (2 * np.pi)
            
            specs.append(torch.stack([mag, phase], dim=0))
        
        return torch.cat(specs, dim=0)


class CGANTrainer:
     
    def __init__(self, generator, discriminator, device='cuda', lr_g=0.0001, lr_d=0.0004):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Otimizadores com betas diferentes
        self.opt_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Schedulers para reduzir LR ao longo do tempo
        self.scheduler_G = optim.lr_scheduler.StepLR(self.opt_G, step_size=10, gamma=0.8)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.opt_D, step_size=10, gamma=0.8)
        
        # Loss functions
        self.criterion_GAN = nn.BCEWithLogitsLoss() 
        self.criterion_L1 = nn.L1Loss()
        self.lambda_L1 = 50  
        
        # Contadores para balanceamento
        self.d_steps = 0
        self.g_steps = 0
    
    def gradient_penalty(self, real, fake):
        """Calcula gradient penalty (WGAN-GP inspired)"""
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        
        interpolates = alpha * real + (1 - alpha) * fake
        interpolates = interpolates.requires_grad_(True)
        
        # Forward pass no discriminador
        mixture = real[:, :4, :, :]  # primeiros 4 canais
        d_interpolates = self.discriminator(mixture, interpolates[:, 4:, :, :])
        
        # Calcula gradientes
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, mixture, target):
        """Passo de treinamento balanceado"""
        mixture = mixture.to(self.device)
        target = target.to(self.device)
        batch_size = mixture.size(0)
        
        # Label smoothing para estabilidade
        real_label_val = 0.9  # ao inves de 1.0
        fake_label_val = 0.1  # ao inves de 0.0
        
        # ============================================
        # Treinar Discriminador
        # ============================================
        d_iters = 3 if self.d_steps < 100 else 1  # Treina mais no inicio
        
        for _ in range(d_iters):
            self.opt_D.zero_grad()
            
            # Gera fake
            with torch.no_grad():
                fake_stem = self.generator(mixture)
            
            # Real
            pred_real = self.discriminator(mixture, target)
            real_label = torch.full_like(pred_real, real_label_val).to(self.device)
            loss_D_real = self.criterion_GAN(pred_real, real_label)
            
            # Fake
            pred_fake = self.discriminator(mixture, fake_stem)
            fake_label = torch.full_like(pred_fake, fake_label_val).to(self.device)
            loss_D_fake = self.criterion_GAN(pred_fake, fake_label)
            
            # Loss total
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            loss_D.backward()
            # Gradient clipping para estabilidade
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.opt_D.step()
            
            self.d_steps += 1
        
        # ============================================
        # Treinar Gerador
        # ============================================
        self.opt_G.zero_grad()
        
        fake_stem = self.generator(mixture)
        
        # GAN loss
        pred_fake = self.discriminator(mixture, fake_stem)
        real_label = torch.ones_like(pred_fake).to(self.device)
        loss_G_GAN = self.criterion_GAN(pred_fake, real_label)
        
        # L1 loss
        loss_G_L1 = self.criterion_L1(fake_stem, target)
        
        # Perceptual loss (diferença no dominio espectral)
        loss_G_perceptual = self._perceptual_loss(fake_stem, target)
        
        # Loss total do gerador
        loss_G = loss_G_GAN + self.lambda_L1 * loss_G_L1 + 10 * loss_G_perceptual
        
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.opt_G.step()
        
        self.g_steps += 1
        
        return {
            'loss_G': loss_G.item(),
            'loss_G_GAN': loss_G_GAN.item(),
            'loss_G_L1': loss_G_L1.item(),
            'loss_G_perceptual': loss_G_perceptual.item(),
            'loss_D': loss_D.item(),
            'loss_D_real': loss_D_real.item(),
            'loss_D_fake': loss_D_fake.item()
        }
    
    def _perceptual_loss(self, pred, target):
        """Loss perceptual no dominio espectral"""
        # Diferença de magnitude media
        pred_mag = pred[:, [0, 2], :, :]  # canais de magnitude
        target_mag = target[:, [0, 2], :, :]
        
        return torch.nn.functional.l1_loss(pred_mag, target_mag)
    
    def step_schedulers(self):
        """Atualiza learning rates"""
        self.scheduler_G.step()
        self.scheduler_D.step()


def compute_sdr(estimate, reference, epsilon=1e-8):
    """SDR"""
    noise = reference - estimate
    sdr = 10 * torch.log10(
        (reference ** 2).sum() / ((noise ** 2).sum() + epsilon)
    )
    return sdr.item()


def validate(generator, val_loader, device):
    """Validacao"""
    generator.eval()
    total_sdr = 0
    total_l1 = 0
    num_batches = 0
    
    criterion_L1 = nn.L1Loss()
    
    with torch.no_grad():
        for mixture, target in tqdm(val_loader, desc='Validacao', leave=False):
            mixture = mixture.to(device)
            target = target.to(device)
            
            prediction = generator(mixture)
            
            l1_loss = criterion_L1(prediction, target).item()
            
            batch_sdr = 0
            for i in range(prediction.size(0)):
                sdr = compute_sdr(prediction[i], target[i])
                batch_sdr += sdr
            
            total_sdr += batch_sdr / prediction.size(0)
            total_l1 += l1_loss
            num_batches += 1
    
    generator.train()
    
    return {
        'avg_sdr': total_sdr / num_batches,
        'avg_l1': total_l1 / num_batches
    }


def main():
    # Constantes de Configuracao
    MUSDB_ROOT = './musdb18'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1
    NUM_EPOCHS = 100  
    LR_G = 0.0001  
    LR_D = 0.0004  
    TARGET_STEM = 'vocals'
    SAVE_EVERY = 2
    
    print(f"Dispositivo: {DEVICE}")
    print(f"Stem alvo: {TARGET_STEM}")
    print(f"LR Generator: {LR_G}, LR Discriminator: {LR_D}")
    
    # Datasets
    print("\nCarregando datasets...")
    train_dataset = MUSDB18Dataset(
        MUSDB_ROOT, 
        subset='train', 
        target_stem=TARGET_STEM,
        use_wav=False
    )
    val_dataset = MUSDB18Dataset(
        MUSDB_ROOT, 
        subset='test', 
        target_stem=TARGET_STEM,
        use_wav=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # Modelo
    generator = UNetGenerator(in_channels=4, out_channels=4)
    discriminator = Discriminator(in_channels=8)
    
    # Trainer
    trainer = CGANTrainer(generator, discriminator, device=DEVICE, 
                                  lr_g=LR_G, lr_d=LR_D)
    
    # Treinamento
    print("\nIniciando treinamento...")
    best_sdr = float('-inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoca {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Treinamento
        epoch_losses = {
            'loss_G': 0, 'loss_G_GAN': 0, 'loss_G_L1': 0, 
            'loss_G_perceptual': 0, 'loss_D': 0,
            'loss_D_real': 0, 'loss_D_fake': 0
        }
        
        for mixture, target in tqdm(train_loader, desc='Treinamento'):
            losses = trainer.train_step(mixture, target)
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        # Medias
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        print(f"\nLosses de Treinamento:")
        print(f"  Generator Total: {epoch_losses['loss_G']:.4f}")
        print(f"  Generator GAN: {epoch_losses['loss_G_GAN']:.4f}")
        print(f"  Generator L1: {epoch_losses['loss_G_L1']:.4f}")
        print(f"  Generator Perceptual: {epoch_losses['loss_G_perceptual']:.4f}")
        print(f"  Discriminator: {epoch_losses['loss_D']:.4f}")
        print(f"    D Real: {epoch_losses['loss_D_real']:.4f}")
        print(f"    D Fake: {epoch_losses['loss_D_fake']:.4f}")
        
        # Validacao
        if (epoch + 1) % SAVE_EVERY == 0:
            val_metrics = validate(generator, val_loader, DEVICE)
            print(f"\nMetricas de Validacao:")
            print(f"  SDR Medio: {val_metrics['avg_sdr']:.2f} dB")
            print(f"  L1 Medio: {val_metrics['avg_l1']:.4f}")
            
            # Early stopping
            if val_metrics['avg_sdr'] > best_sdr:
                best_sdr = val_metrics['avg_sdr']
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'sdr': best_sdr
                }, f'best_model_{TARGET_STEM}.pth')
                print(f"Melhor modelo salvo - SDR: {best_sdr:.2f} dB")
            else:
                patience_counter += 1
                print(f"SDR nao melhorou ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print("\nEarly stopping acionado")
                    break
        
        # Atualiza learning rates
        trainer.step_schedulers()
        print(f"  LR atual - G: {trainer.opt_G.param_groups[0]['lr']:.6f}, "
              f"D: {trainer.opt_D.param_groups[0]['lr']:.6f}")
    
    print("\n" + "="*60)
    print("Treinamento concluido")
    print(f"Melhor SDR: {best_sdr:.2f} dB")
    print("="*60)


if __name__ == '__main__':
    main()