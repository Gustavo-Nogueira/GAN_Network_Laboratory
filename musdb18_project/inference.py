import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from train import UNetGenerator

class StemSeparator:
    """Classe para inferencia com modelo"""
    
    def __init__(self, model_path, device='cuda', n_fft=2048, hop_length=512):
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft).to(self.device)
        
        # Carrega modelo
        print(f"Carregando modelo de: {model_path}")
        self.generator = UNetGenerator(in_channels=4, out_channels=4)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.to(self.device)
        self.generator.eval()
        
        print(f"Modelo carregado (Epoca: {checkpoint.get('epoch', 'N/A')}, "
              f"SDR: {checkpoint.get('sdr', 'N/A'):.2f} dB)")
        print(f"Dispositivo: {self.device}")
    
    def load_audio(self, audio_path, target_sr=44100):
        """Carrega arquivo de audio"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample se necessario
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Converte para stereo se mono
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]
        
        return waveform, target_sr
    
    def audio_to_spectrogram(self, audio):
        """
        Converte audio para espectrograma normalizado
        Usa mesma normalizacao do treinamento
        """
        # Normaliza audio primeiro
        audio = audio / (audio.abs().max() + 1e-8)
        audio = audio.to(self.device)
        
        specs = []
        
        for ch in range(audio.shape[0]):
            # STFT
            spec = torch.stft(
                audio[ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            
            # Magnitude e fase
            mag = torch.abs(spec)
            phase = torch.angle(spec)
            
            # Normalizacao (igual ao treinamento)
            mag = mag / (mag.max() + 1e-8)  # [0, 1]
            phase = (phase + np.pi) / (2 * np.pi)  # [0, 1]
            
            specs.append(torch.stack([mag, phase], dim=0))
        
        # Concatena canais: (4, freq, time)
        spectrogram = torch.cat(specs, dim=0)
        return spectrogram
    
    def spectrogram_to_audio(self, spectrogram, original_length=None):
        """
        Converte espectrograma de volta para audio
        """
        # Separa magnitude e fase para cada canal
        mag_left = spectrogram[0]
        phase_left = spectrogram[1]
        mag_right = spectrogram[2]
        phase_right = spectrogram[3]
        
        # Desnormalizacao
        # Magnitude ja esta em [0, 1], mas precisamos de valores reais
        phase_left = phase_left * (2 * np.pi) - np.pi
        phase_right = phase_right * (2 * np.pi) - np.pi
        
        # Reconstroi complexo
        spec_left = mag_left * torch.exp(1j * phase_left)
        spec_right = mag_right * torch.exp(1j * phase_right)
        
        # iSTFT
        audio_left = torch.istft(
            spec_left,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=original_length
        )
        audio_right = torch.istft(
            spec_right,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=original_length
        )
        
        # Junta canais
        audio = torch.stack([audio_left, audio_right], dim=0)
        return audio
    
    def separate_stem(self, audio_path, output_path=None, chunk_duration=30.0):
        """
        Separa stem de um arquivo de audio
        """

        print(f"\nProcessando: {audio_path}")
        
        # Carrega audio
        waveform, sr = self.load_audio(audio_path)
        original_length = waveform.shape[1]
        print(f"audio carregado: {original_length/sr:.2f}s")
        
        # Divide em chunks se necessario
        chunk_size = int(chunk_duration * sr)
        total_samples = waveform.shape[1]
        
        if total_samples > chunk_size:
            print(f"Dividindo em chunks de {chunk_duration}s...")
            separated_chunks = []
            
            for start in tqdm(range(0, total_samples, chunk_size), desc="Processando"):
                end = min(start + chunk_size, total_samples)
                chunk = waveform[:, start:end]
                chunk_length = chunk.shape[1]
                
                # Processa chunk
                sep_chunk = self._process_chunk(chunk, original_length=chunk_length)
                separated_chunks.append(sep_chunk)
            
            # Junta chunks
            separated_stem = torch.cat(separated_chunks, dim=1)
        else:
            separated_stem = self._process_chunk(waveform, original_length=total_samples)
        
        # Trunca para o tamanho original
        separated_stem = separated_stem[:, :original_length]
        
        print(f"Separacao concluida")
        
        # Salva se output_path fornecido
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Normaliza para evitar clipping
            max_val = separated_stem.abs().max()
            if max_val > 0:
                separated_stem_normalized = separated_stem / (max_val + 1e-8)
                # Aplica soft limiting para evitar clipping
                separated_stem_normalized = torch.tanh(separated_stem_normalized * 0.95) * 0.95
            else:
                separated_stem_normalized = separated_stem
            
            torchaudio.save(str(output_path), separated_stem_normalized.cpu(), sr)
            print(f"Salvo em: {output_path}")
        
        return separated_stem, sr
    
    def _process_chunk(self, audio_chunk, original_length=None):
        """Processa um chunk de audio"""
        # Converte para espectrograma
        spec = self.audio_to_spectrogram(audio_chunk)
        
        # Adiciona batch dimension
        spec = spec.unsqueeze(0)  # (1, 4, freq, time)
        
        # Inferencia
        with torch.no_grad():
            separated_spec = self.generator(spec)
        
        # Remove batch dimension
        separated_spec = separated_spec.squeeze(0)  # (4, freq, time)
        
        # Converte de volta para audio
        separated_audio = self.spectrogram_to_audio(separated_spec, original_length)
        
        return separated_audio

def main():
    # Constantes
    N_FFT = 2048
    HOP_LENGTH = 512
    CHUNK_DURATION = 30.0

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    separator = StemSeparator(
        model_path=args.model,
        device=args.device,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
               
    output_path = args.output
    separator.separate_stem(args.input, output_path, chunk_duration=CHUNK_DURATION)

if __name__ == '__main__':  
    main()