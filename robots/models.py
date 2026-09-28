"""The small networks both robot studies train, and the way motions are cut for them.

Every model reads a short window of human joint positions and writes the same window of
robot joint positions. The window is 64 frames at 30 frames per second, a little over two
seconds. Positions are taken relative to the root joint and divided by one scale per
skeleton, so the two sides are comparable in size.

Two shapes of model appear in the paper:

DirectMap        one transformer that turns a human window straight into a robot window.
                 What it learns depends only on what it is trained against, which is how
                 the averaging objective, the true-pair model and the adversarial model
                 differ from one another.
AutoencoderPair  one autoencoder per skeleton. An autoencoder is a network that squeezes
                 each window into a short code and rebuilds the window from that code; the
                 codes it can produce form its latent space. The two autoencoders share one
                 latent space, but neither ever sees the other skeleton, so the
                 human-to-robot map is whatever falls out of that shared space: encode the
                 human window, decode the code as a robot window.

The adversarial model also uses a critic: a second network trained to tell real robot
windows from produced ones, while the model is trained to make it fail.
"""
import numpy as np
import torch
import torch.nn as nn

FRAMES = 64


def root_center(clip):
    """Move each frame so the root joint sits at the origin."""
    return clip - clip[:, 0:1, :]


def split_windows(clip, frames=FRAMES):
    """Cut a motion into back-to-back windows, dropping any frames left over at the end.

    A motion shorter than one window is returned as a single window padded with zeros.
    """
    length = clip.shape[0]
    if length < frames:
        padded = np.zeros((frames,) + clip.shape[1:], clip.dtype)
        padded[:length] = clip
        return padded[None]
    count = length // frames
    return np.stack([clip[i * frames:(i + 1) * frames] for i in range(count)])


def windows_per_clip(clips, side, frames=FRAMES):
    """Windows for every clip on one side, plus the single scale used to normalise them."""
    per_clip, flat = {}, []
    for clip_id, clip in clips.items():
        w = split_windows(root_center(clip[side].astype(np.float32)), frames)
        per_clip[clip_id] = w
        flat.append(w.reshape(-1))
    return per_clip, float(np.concatenate(flat).std()) + 1e-6


class MotionTransformer(nn.Module):
    """A short transformer over the frames of one window, with learned frame positions."""

    def __init__(self, width=256, layers=4, heads=8, frames=FRAMES):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(frames, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        return self.enc(x + self.pos.unsqueeze(0))


class DirectMap(nn.Module):
    """Human window (batch, frames, source joints, 3) to robot window, in one pass."""

    def __init__(self, source_joints, target_joints, width=256):
        super().__init__()
        self.source_joints, self.target_joints = source_joints, target_joints
        self.proj_in = nn.Linear(source_joints * 3, width)
        self.backbone = MotionTransformer(width)
        self.proj_out = nn.Linear(width, target_joints * 3)

    def forward(self, x):
        batch, frames = x.shape[:2]
        h = self.proj_in(x.reshape(batch, frames, self.source_joints * 3))
        h = self.backbone(h)
        return self.proj_out(h).reshape(batch, frames, self.target_joints, 3)


class Autoencoder(nn.Module):
    """One skeleton's own encoder and decoder, with a single latent code per window."""

    def __init__(self, joints, width=256, latent=64):
        super().__init__()
        self.joints = joints
        self.proj_in = nn.Linear(joints * 3, width)
        self.enc = MotionTransformer(width)
        self.to_z = nn.Linear(width, 2 * latent)
        self.from_z = nn.Linear(latent, width)
        self.dec = MotionTransformer(width)
        self.proj_out = nn.Linear(width, joints * 3)

    def encode(self, x):
        batch, frames = x.shape[:2]
        h = self.enc(self.proj_in(x.reshape(batch, frames, self.joints * 3)))
        mean, log_var = self.to_z(h.mean(1)).chunk(2, dim=-1)
        return mean, log_var

    def decode(self, z, frames=FRAMES):
        h = self.from_z(z).unsqueeze(1).expand(z.shape[0], frames, -1)
        h = self.dec(h)
        return self.proj_out(h).reshape(z.shape[0], frames, self.joints, 3)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        return self.decode(z, x.shape[1]), mean, log_var


class AutoencoderPair(nn.Module):
    """Two autoencoders sharing one latent space; the map is encode human, decode robot."""

    def __init__(self, source_joints, target_joints, width=256, latent=64):
        super().__init__()
        self.human = Autoencoder(source_joints, width, latent)
        self.robot = Autoencoder(target_joints, width, latent)

    def cross(self, human):
        mean, _ = self.human.encode(human)
        return self.robot.decode(mean, human.shape[1])


class Critic(nn.Module):
    """Tells real robot windows from produced ones; used by the adversarial objective."""

    def __init__(self, joints, width=256):
        super().__init__()
        self.joints = joints
        self.proj_in = nn.Linear(joints * 3, width)
        self.backbone = MotionTransformer(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        batch, frames = x.shape[:2]
        h = self.proj_in(x.reshape(batch, frames, self.joints * 3))
        h = self.backbone(h)
        return self.head(h.mean(1)).squeeze(-1)


def kl_divergence(mean, log_var):
    """How far a window's latent code sits from the shared normal prior."""
    return (-0.5 * (1 + log_var - mean.pow(2) - log_var.exp())).sum(-1).mean()
