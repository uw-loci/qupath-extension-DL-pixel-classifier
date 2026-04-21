"""Pretrained models service for listing available architectures and layer structures."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EncoderInfo:
    """Information about an available encoder."""
    name: str
    display_name: str
    family: str
    params_millions: float
    pretrained_weights: List[str]
    license: str
    recommended_for: List[str] = field(default_factory=list)


@dataclass
class ArchitectureInfo:
    """Information about a segmentation architecture."""
    name: str
    display_name: str
    description: str
    decoder_channels: List[int]
    supports_aux_output: bool = False


@dataclass
class LayerInfo:
    """Information about a freezable layer."""
    name: str
    display_name: str
    param_count: int
    is_encoder: bool
    depth: int  # 0 = earliest/most general, higher = deeper/more specific
    recommended_freeze: bool  # Whether to freeze by default for fine-tuning


class PretrainedModelsService:
    """Service for managing pretrained model architectures and encoders."""

    # Histology-pretrained encoders: composite_name -> (smp_encoder_name, timm_hub_id)
    # These use ResNet-50 architecture but with weights pretrained on histopathology data
    # instead of ImageNet, providing much better feature extraction for tissue classification.
    HISTOLOGY_ENCODERS = {
        "resnet50_lunit-swav": ("resnet50", "hf_hub:1aurent/resnet50.lunit_swav"),
        "resnet50_lunit-bt": ("resnet50", "hf_hub:1aurent/resnet50.lunit_bt"),
        "resnet50_kather100k": ("resnet50", "hf_hub:1aurent/resnet50.tiatoolbox-kather100k"),
        "resnet50_tcga-brca": ("resnet50", "hf_hub:1aurent/resnet50.tcga_brca_simclr"),
    }

    # Foundation model encoders: downloaded on-demand from HuggingFace.
    # Only models with commercially-permissive licenses (Apache 2.0, MIT, CC-BY-4.0) are included.
    # Integration approach inspired by LazySlide (MIT License).
    # Zheng, Y. et al. Nature Methods (2026). https://doi.org/10.1038/s41592-026-03044-7
    FOUNDATION_ENCODERS = {
        # Apache 2.0 - Bioptimus, 1.1B params, ViT-g, 1536-dim, gated access
        "h-optimus-0": ("timm_foundation", "hf_hub:bioptimus/H-optimus-0"),
        # Apache 2.0 - Paige AI, 632M params, ViT-H, 2560-dim, gated access
        "virchow": ("timm_foundation", "hf_hub:paige-ai/Virchow"),
        # Apache 2.0 - HistAI, ~300M params, ViT-L/14, 1024-dim, gated access
        "hibou-l": ("timm_foundation", "hf_hub:histai/hibou-L"),
        # Apache 2.0 - HistAI, 85.7M params, ViT-B/14, 768-dim, gated access
        "hibou-b": ("timm_foundation", "hf_hub:histai/hibou-b"),
        # MIT - Kaiko AI, ViT-G, 1536-dim, ungated (TCGA-only training)
        "midnight": ("timm_foundation", "hf_hub:kaiko-ai/midnight"),
        # Apache 2.0 - Meta, ViT-L/14, 1024-dim, ungated (general-purpose, not pathology-specific)
        "dinov2-large": ("timm_foundation", "hf_hub:facebook/dinov2-large"),
    }

    def __init__(self):
        self._encoders = self._init_encoders()
        self._architectures = self._init_architectures()

    def _init_encoders(self) -> Dict[str, EncoderInfo]:
        """Initialize available encoders from segmentation-models-pytorch."""
        # These are the most commonly used and well-tested encoders for histopathology
        return {
            # ResNet family - good general purpose, well-tested
            "resnet18": EncoderInfo(
                name="resnet18", display_name="ResNet-18",
                family="resnet", params_millions=11.7,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["small_datasets", "fast_inference"]
            ),
            "resnet34": EncoderInfo(
                name="resnet34", display_name="ResNet-34",
                family="resnet", params_millions=21.8,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["general_purpose", "balanced"]
            ),
            "resnet50": EncoderInfo(
                name="resnet50", display_name="ResNet-50",
                family="resnet", params_millions=25.6,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["large_datasets", "high_accuracy"]
            ),

            # Histology-pretrained ResNet-50 encoders
            # These use the same ResNet-50 architecture but with weights pretrained
            # on millions of histopathology patches, providing much better feature
            # extraction for tissue classification than ImageNet weights.
            "resnet50_lunit-swav": EncoderInfo(
                name="resnet50_lunit-swav",
                display_name="ResNet-50 Lunit SwAV (Histology)",
                family="resnet", params_millions=25.6,
                pretrained_weights=["histology"],
                license="Non-commercial (Lunit)",
                recommended_for=["histopathology", "high_accuracy", "tissue_classification"]
            ),
            "resnet50_lunit-bt": EncoderInfo(
                name="resnet50_lunit-bt",
                display_name="ResNet-50 Lunit Barlow Twins (Histology)",
                family="resnet", params_millions=25.6,
                pretrained_weights=["histology"],
                license="Non-commercial (Lunit)",
                recommended_for=["histopathology", "high_accuracy", "tissue_classification"]
            ),
            "resnet50_kather100k": EncoderInfo(
                name="resnet50_kather100k",
                display_name="ResNet-50 Kather100K (Histology)",
                family="resnet", params_millions=25.6,
                pretrained_weights=["histology"],
                license="CC-BY-4.0",
                recommended_for=["histopathology", "colorectal", "tissue_classification"]
            ),
            "resnet50_tcga-brca": EncoderInfo(
                name="resnet50_tcga-brca",
                display_name="ResNet-50 TCGA-BRCA (Histology)",
                family="resnet", params_millions=25.6,
                pretrained_weights=["histology"],
                license="GPLv3",
                recommended_for=["histopathology", "breast_cancer", "tissue_classification"]
            ),

            "resnet101": EncoderInfo(
                name="resnet101", display_name="ResNet-101",
                family="resnet", params_millions=44.5,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["very_large_datasets"]
            ),

            # EfficientNet family - good accuracy/speed tradeoff
            "efficientnet-b0": EncoderInfo(
                name="efficientnet-b0", display_name="EfficientNet-B0",
                family="efficientnet", params_millions=5.3,
                pretrained_weights=["imagenet"],
                license="Apache 2.0",
                recommended_for=["efficient", "mobile"]
            ),
            "efficientnet-b3": EncoderInfo(
                name="efficientnet-b3", display_name="EfficientNet-B3",
                family="efficientnet", params_millions=12.0,
                pretrained_weights=["imagenet"],
                license="Apache 2.0",
                recommended_for=["balanced", "efficient"]
            ),
            "efficientnet-b4": EncoderInfo(
                name="efficientnet-b4", display_name="EfficientNet-B4",
                family="efficientnet", params_millions=19.0,
                pretrained_weights=["imagenet"],
                license="Apache 2.0",
                recommended_for=["high_accuracy", "efficient"]
            ),

            # SE-ResNet family - attention mechanism
            "se_resnet50": EncoderInfo(
                name="se_resnet50", display_name="SE-ResNet-50",
                family="se_resnet", params_millions=28.1,
                pretrained_weights=["imagenet"],
                license="MIT",
                recommended_for=["attention", "histopathology"]
            ),

            # DenseNet family - feature reuse
            "densenet121": EncoderInfo(
                name="densenet121", display_name="DenseNet-121",
                family="densenet", params_millions=8.0,
                pretrained_weights=["imagenet"],
                license="BSD",
                recommended_for=["feature_reuse", "small_objects"]
            ),
            "densenet169": EncoderInfo(
                name="densenet169", display_name="DenseNet-169",
                family="densenet", params_millions=14.1,
                pretrained_weights=["imagenet"],
                license="BSD",
                recommended_for=["feature_reuse"]
            ),

            # MobileNet - lightweight
            "mobilenet_v2": EncoderInfo(
                name="mobilenet_v2", display_name="MobileNet-V2",
                family="mobilenet", params_millions=3.5,
                pretrained_weights=["imagenet"],
                license="Apache 2.0",
                recommended_for=["mobile", "fast_inference", "low_memory"]
            ),

            # VGG - classic, good for texture
            "vgg16_bn": EncoderInfo(
                name="vgg16_bn", display_name="VGG-16 (BatchNorm)",
                family="vgg", params_millions=138.4,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["texture", "histopathology"]
            ),
        }

    def _init_architectures(self) -> Dict[str, ArchitectureInfo]:
        """Initialize available segmentation architectures."""
        return {
            "unet": ArchitectureInfo(
                name="unet", display_name="U-Net",
                description="Classic encoder-decoder with skip connections. Best general-purpose choice.",
                decoder_channels=[256, 128, 64, 32, 16]
            ),
            "unetplusplus": ArchitectureInfo(
                name="unetplusplus", display_name="U-Net++",
                description="Nested U-Net with dense skip connections. Better for small objects.",
                decoder_channels=[256, 128, 64, 32, 16],
                supports_aux_output=True
            ),
            "deeplabv3": ArchitectureInfo(
                name="deeplabv3", display_name="DeepLab V3",
                description="Atrous convolution for multi-scale context. Good for large structures.",
                decoder_channels=[256]
            ),
            "deeplabv3plus": ArchitectureInfo(
                name="deeplabv3plus", display_name="DeepLab V3+",
                description="DeepLab V3 with decoder. Better boundary delineation.",
                decoder_channels=[256, 48]
            ),
            "fpn": ArchitectureInfo(
                name="fpn", display_name="Feature Pyramid Network",
                description="Multi-scale feature pyramid. Good for varying object sizes.",
                decoder_channels=[256, 256, 256, 256]
            ),
            "pspnet": ArchitectureInfo(
                name="pspnet", display_name="PSPNet",
                description="Pyramid pooling module for global context.",
                decoder_channels=[512]
            ),
            "manet": ArchitectureInfo(
                name="manet", display_name="MA-Net",
                description="Multi-scale attention network. Good for complex boundaries.",
                decoder_channels=[256, 128, 64, 32, 16]
            ),
            "linknet": ArchitectureInfo(
                name="linknet", display_name="LinkNet",
                description="Lightweight encoder-decoder. Fast inference.",
                decoder_channels=[256, 128, 64, 32]
            ),
        }

    def list_encoders(self) -> List[Dict[str, Any]]:
        """List available encoders."""
        return [
            {
                "name": e.name,
                "display_name": e.display_name,
                "family": e.family,
                "params_millions": e.params_millions,
                "pretrained_weights": e.pretrained_weights,
                "license": e.license,
                "recommended_for": e.recommended_for
            }
            for e in self._encoders.values()
        ]

    def list_architectures(self) -> List[Dict[str, Any]]:
        """List available segmentation architectures."""
        return [
            {
                "name": a.name,
                "display_name": a.display_name,
                "description": a.description,
                "decoder_channels": a.decoder_channels,
                "supports_aux_output": a.supports_aux_output
            }
            for a in self._architectures.values()
        ]

    def get_model_layers(
        self,
        architecture: str,
        encoder: str,
        num_channels: int = 3,
        num_classes: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get the layer structure of a model for freeze/unfreeze configuration.

        Returns a list of layer groups that can be frozen/unfrozen.
        """
        try:
            import segmentation_models_pytorch as smp
            import torch

            # Create model to inspect structure
            model = self._create_model(architecture, encoder, num_channels, num_classes)

            layers = []

            # Get encoder layers
            encoder_layers = self._get_encoder_layers(model, encoder, num_channels)
            layers.extend(encoder_layers)

            # Get decoder layers
            decoder_layers = self._get_decoder_layers(model, architecture)
            layers.extend(decoder_layers)

            return layers

        except ImportError:
            logger.error("segmentation_models_pytorch not installed")
            return []
        except Exception as e:
            # exc_info=True so stack trace is logged -- otherwise bugs like
            # missing-arg or free-variable references are invisible and the
            # Java side silently falls back to built-in layer defaults.
            logger.error("Error getting model layers: %s", e, exc_info=True)
            return []

    def _create_model(
        self,
        architecture: str,
        encoder: str,
        num_channels: int,
        num_classes: int
    ):
        """Create a segmentation model for inspection."""
        import segmentation_models_pytorch as smp

        arch_map = {
            "unet": smp.Unet,
            "fast-pretrained": smp.Unet,  # UNet with mobile encoder
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "manet": smp.MAnet,
            "linknet": smp.Linknet,
        }

        if architecture not in arch_map:
            raise ValueError(f"Unknown architecture: {architecture}")

        # For histology encoders, resolve the smp encoder name and load
        # the model with imagenet weights first (correct architecture),
        # then replace encoder weights with histology-pretrained weights.
        if encoder in self.HISTOLOGY_ENCODERS:
            smp_encoder, hub_id = self.HISTOLOGY_ENCODERS[encoder]
            model = arch_map[architecture](
                encoder_name=smp_encoder,
                encoder_weights="imagenet",
                in_channels=num_channels,
                classes=num_classes
            )
            self._load_histology_weights(model, hub_id, smp_encoder, num_channels)
            return model

        return arch_map[architecture](
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=num_channels,
            classes=num_classes
        )

    def _load_histology_weights(self, model, hub_id: str, encoder_name: str,
                                num_channels: int = 3):
        """
        Load histology-pretrained weights from HuggingFace via timm into smp encoder.

        Creates a timm model with the specified hub_id, then transfers matching
        state_dict keys to the smp model's encoder. Skips the first conv layer
        if num_channels != 3 to preserve smp's channel adaptation.

        Args:
            model: smp segmentation model with encoder to update
            hub_id: HuggingFace/timm model identifier
            encoder_name: smp encoder name (for logging)
            num_channels: number of input channels
        """
        try:
            import timm

            logger.info("Downloading histology encoder weights: %s "
                        "(~100MB on first use, cached in ~/.cache/huggingface/)",
                        hub_id)

            # Load the timm model with pretrained weights from HuggingFace
            timm_model = timm.create_model(hub_id, pretrained=True)
            timm_state = timm_model.state_dict()

            # Get smp encoder state dict
            encoder_state = model.encoder.state_dict()

            # Transfer matching keys from timm to smp encoder
            loaded_count = 0
            total_count = len(encoder_state)

            for key in encoder_state:
                if key in timm_state:
                    # Skip first conv layer if channel count differs from 3
                    # to preserve smp's automatic channel adaptation
                    if num_channels != 3 and ("conv1.weight" in key or
                                               "features.0.weight" in key or
                                               "_conv_stem.weight" in key):
                        logger.debug("Skipping %s (num_channels=%d != 3)", key,
                                     num_channels)
                        continue

                    if encoder_state[key].shape == timm_state[key].shape:
                        encoder_state[key] = timm_state[key]
                        loaded_count += 1
                    else:
                        logger.debug("Shape mismatch for %s: smp=%s, timm=%s",
                                     key, encoder_state[key].shape,
                                     timm_state[key].shape)

            model.encoder.load_state_dict(encoder_state)
            logger.info("Loaded %d/%d histology encoder weights from %s",
                        loaded_count, total_count, hub_id)

            if loaded_count < total_count * 0.5:
                logger.warning("Less than 50%% of encoder weights matched for %s. "
                               "The model may not benefit from histology pretraining.",
                               hub_id)

            # Free timm model memory
            del timm_model, timm_state

        except ImportError:
            raise RuntimeError(
                "The 'timm' package is required for histology-pretrained encoders. "
                "Install it with: pip install timm>=1.0.0"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load histology encoder weights from {hub_id}: {e}. "
                "Check your internet connection and try again. "
                "The weights (~100MB) are downloaded from HuggingFace on first use."
            )

    def _get_encoder_layers(self, model, encoder_name: str,
                            num_channels: int = 3) -> List[Dict[str, Any]]:
        """Extract encoder layer information."""
        layers = []

        if hasattr(model, 'encoder'):
            encoder = model.encoder

            # Different encoder families have different structures
            encoder_family = self._encoders.get(encoder_name, EncoderInfo(
                name=encoder_name, display_name=encoder_name,
                family="unknown", params_millions=0,
                pretrained_weights=[], license=""
            )).family

            if encoder_family in ["resnet", "se_resnet"]:
                # ResNet-style: conv1 -> layer1 -> layer2 -> layer3 -> layer4
                layer_names = [
                    ("encoder.conv1", "Initial Conv", 0, True),  # Freeze by default
                    ("encoder.layer1", "Block 1 (64 filters)", 1, True),
                    ("encoder.layer2", "Block 2 (128 filters)", 2, True),
                    ("encoder.layer3", "Block 3 (256 filters)", 3, False),
                    ("encoder.layer4", "Block 4 (512 filters)", 4, False),
                ]
            elif encoder_family == "efficientnet":
                # EfficientNet-style
                layer_names = [
                    ("encoder._conv_stem", "Stem Conv", 0, True),
                    ("encoder._blocks[0:4]", "Blocks 0-3", 1, True),
                    ("encoder._blocks[4:10]", "Blocks 4-9", 2, True),
                    ("encoder._blocks[10:18]", "Blocks 10-17", 3, False),
                    ("encoder._blocks[18:]", "Blocks 18+", 4, False),
                ]
            elif encoder_family == "densenet":
                layer_names = [
                    ("encoder.features.conv0", "Initial Conv", 0, True),
                    ("encoder.features.denseblock1", "Dense Block 1", 1, True),
                    ("encoder.features.denseblock2", "Dense Block 2", 2, True),
                    ("encoder.features.denseblock3", "Dense Block 3", 3, False),
                    ("encoder.features.denseblock4", "Dense Block 4", 4, False),
                ]
            elif encoder_family == "vgg":
                layer_names = [
                    ("encoder.features[0:7]", "Layers 1-2 (64 filters)", 0, True),
                    ("encoder.features[7:14]", "Layers 3-4 (128 filters)", 1, True),
                    ("encoder.features[14:24]", "Layers 5-7 (256 filters)", 2, True),
                    ("encoder.features[24:34]", "Layers 8-10 (512 filters)", 3, False),
                    ("encoder.features[34:]", "Layers 11-13 (512 filters)", 4, False),
                ]
            elif encoder_family == "mobilenet":
                layer_names = [
                    ("encoder.features[0:2]", "Initial Conv", 0, True),
                    ("encoder.features[2:5]", "Blocks 1-3", 1, True),
                    ("encoder.features[5:9]", "Blocks 4-7", 2, True),
                    ("encoder.features[9:14]", "Blocks 8-12", 3, False),
                    ("encoder.features[14:]", "Blocks 13+", 4, False),
                ]
            else:
                # Generic fallback
                layer_names = [
                    ("encoder.layer_early", "Early Layers", 0, True),
                    ("encoder.layer_mid", "Middle Layers", 2, False),
                    ("encoder.layer_late", "Late Layers", 4, False),
                ]

            # Count parameters for each layer group
            for name, display_name, depth, freeze_default in layer_names:
                param_count = self._count_params_for_layer(encoder, name)
                # When input channels differ from pretrained (3), the
                # depth-0 layer contains adapted (not truly pretrained)
                # weights and should NOT be frozen by default.
                rec_freeze = freeze_default
                desc = self._get_layer_description(depth, True, encoder_name)
                if depth == 0 and num_channels != 3:
                    rec_freeze = False
                    desc = (
                        "Contains channel-adapted weights (input has "
                        f"{num_channels}ch, pretrained on 3ch). "
                        "Should be trainable so the model can learn "
                        "scale-specific features."
                    )
                layers.append({
                    "name": name,
                    "display_name": f"Encoder: {display_name}",
                    "param_count": param_count,
                    "is_encoder": True,
                    "depth": depth,
                    "recommended_freeze": rec_freeze,
                    "description": desc
                })

        return layers

    def _get_decoder_layers(self, model, architecture: str) -> List[Dict[str, Any]]:
        """Extract decoder layer information."""
        layers = []

        if hasattr(model, 'decoder'):
            decoder = model.decoder
            param_count = sum(p.numel() for p in decoder.parameters())

            layers.append({
                "name": "decoder",
                "display_name": "Decoder (all layers)",
                "param_count": param_count,
                "is_encoder": False,
                "depth": 5,
                "recommended_freeze": False,  # Never freeze decoder for fine-tuning
                "description": "Task-specific layers - should always be trained"
            })

        if hasattr(model, 'segmentation_head'):
            head = model.segmentation_head
            param_count = sum(p.numel() for p in head.parameters())

            layers.append({
                "name": "segmentation_head",
                "display_name": "Segmentation Head",
                "param_count": param_count,
                "is_encoder": False,
                "depth": 6,
                "recommended_freeze": False,
                "description": "Final classification layer - must be trained"
            })

        return layers

    def _resolve_slice_modules(self, parent_module, part_with_brackets: str):
        """Parse 'attr[start:end]' and return list of sub-modules in the slice range.

        Handles patterns like:
            _blocks[0:4]   -> modules at indices 0, 1, 2, 3
            features[7:14] -> modules at indices 7 through 13
            _blocks[18:]   -> modules at indices 18 to end
            features[0:2]  -> modules at indices 0, 1

        Args:
            parent_module: the parent module containing the attribute
            part_with_brackets: string like "_blocks[0:4]" or "features[18:]"

        Returns:
            list of sub-modules matching the slice, or empty list on failure
        """
        match = re.match(r'^(\w+)\[(\d*):(\d*)\]$', part_with_brackets)
        if not match:
            return []

        attr_name = match.group(1)
        start_str = match.group(2)
        end_str = match.group(3)

        if not hasattr(parent_module, attr_name):
            return []

        container = getattr(parent_module, attr_name)
        children = list(container)

        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else len(children)

        return children[start:end]

    def _count_params_for_layer(self, module, layer_name: str) -> int:
        """Count parameters in a layer group."""
        try:
            parts = layer_name.replace("encoder.", "").split(".")
            current = module

            for part in parts:
                if "[" in part:
                    # Handle slice notation like "features[0:7]" or "_blocks[0:4]"
                    sub_modules = self._resolve_slice_modules(current, part)
                    return sum(
                        p.numel()
                        for m in sub_modules
                        for p in m.parameters()
                    )
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return 0

            return sum(p.numel() for p in current.parameters())
        except Exception:
            return 0

    def _get_layer_description(self, depth: int, is_encoder: bool,
                               encoder: Optional[str] = None) -> str:
        """Get description for a layer based on its depth and what features it captures."""
        if not is_encoder:
            return "Task-specific output layer - always train"

        is_histology = encoder is not None and encoder in self.HISTOLOGY_ENCODERS

        if is_histology:
            # Histology-pretrained encoders already capture tissue-relevant features
            # at all depths, so transfer is much better than ImageNet
            descriptions = {
                0: "Basic tissue textures - already tissue-aware, safe to freeze",
                1: "Cell-level patterns - tissue-relevant, freeze for small datasets",
                2: "Tissue microstructure - already captures histology patterns, consider training",
                3: "Tissue architecture features - already relevant, train for best adaptation",
                4: "High-level tissue semantics - good starting point, fine-tune for your task",
            }
        else:
            # ImageNet-pretrained: significant domain shift to histopathology
            descriptions = {
                0: "Edges, gradients, basic textures - universal features, transfer well, freeze",
                1: "Low-level patterns (gabor-like filters) - transfer well across domains, freeze",
                2: "Texture combinations, local patterns - partial transfer, consider fine-tuning",
                3: "Mid-level shapes, larger patterns - limited transfer to histopathology, train",
                4: "High-level semantic features - ImageNet concepts don't apply, must retrain",
            }

        return descriptions.get(depth, "Deep features - likely need retraining")

    def get_freeze_recommendations(self, dataset_size: str,
                                    encoder: Optional[str] = None
                                    ) -> Dict[str, bool]:
        """
        Get recommended freeze settings based on domain adaptation needs.

        For ImageNet-pretrained encoders, the primary factor is what each layer
        learns and how well it transfers from natural images to histopathology
        (significant domain shift). For histology-pretrained encoders, features
        are already tissue-relevant so less freezing is needed.

        Args:
            dataset_size: "small" (<500 tiles), "medium" (500-5000), "large" (>5000)
                         This affects risk of overfitting when training more layers
            encoder: Optional encoder name. If a histology encoder, returns
                    less aggressive freeze recommendations.

        Returns:
            Dict mapping layer depth to freeze recommendation
        """
        is_histology = encoder is not None and encoder in self.HISTOLOGY_ENCODERS

        if is_histology:
            # Histology encoders already have tissue-relevant features at all
            # depths, so we need much less freezing. Mid-level features (depth
            # 2-3) already capture tissue structures, unlike ImageNet features
            # at those depths which encode natural-image concepts.
            if dataset_size == "small":
                # Small: freeze only earliest layers to prevent overfitting
                return {0: True, 1: True, 2: False, 3: False, 4: False}
            elif dataset_size == "medium":
                # Medium: freeze only the initial conv
                return {0: True, 1: False, 2: False, 3: False, 4: False}
            else:  # large
                # Large: fine-tune everything - histology features are a
                # great starting point but full adaptation gives best results
                return {0: False, 1: False, 2: False, 3: False, 4: False}
        else:
            # ImageNet-pretrained: significant domain shift to histopathology
            # Early layers (0-1): Universal visual features - always freeze
            # Late layers (3-4): Semantic features that don't transfer - always train
            # Middle layers (2): Depends on data available to prevent overfitting
            if dataset_size == "small":
                return {0: True, 1: True, 2: True, 3: True, 4: False}
            elif dataset_size == "medium":
                return {0: True, 1: True, 2: True, 3: False, 4: False}
            else:  # large
                return {0: True, 1: True, 2: False, 3: False, 4: False}

    def create_model_with_frozen_layers(
        self,
        architecture: str,
        encoder: str,
        num_channels: int,
        num_classes: int,
        frozen_layers: List[str]
    ):
        """
        Create a model with specified layers frozen.

        Args:
            architecture: Model architecture name
            encoder: Encoder name
            num_channels: Input channels
            num_classes: Number of output classes
            frozen_layers: List of layer names to freeze

        Returns:
            PyTorch model with frozen layers
        """
        model = self._create_model(architecture, encoder, num_channels, num_classes)
        self.apply_frozen_layers(model, frozen_layers, num_channels)
        return model

    def apply_frozen_layers(self, model, frozen_layers, num_channels: int):
        """Freeze the named layer groups on an already-created model.

        Separate from create_model_with_frozen_layers so callers that build
        non-SMP models (Tiny UNet, MuViT) can still reuse the first-conv
        guard + logging rather than duplicating freeze logic.

        Returns the same model for chaining.
        """
        # Guard: skip freezing the first conv layer when input channels
        # differ from the pretrained channel count (3). SMP adapts conv1
        # by repeating/scaling pretrained weights, so those weights are NOT
        # truly pretrained for the extra channels (e.g. context_scale > 1
        # doubles channels to 6). Freezing them locks in a naive
        # initialization that treats detail and context tiles identically.
        pretrained_channels = 3
        first_conv_names = ("encoder.conv1", "encoder._conv_stem",
                            "encoder.features.conv0", "encoder.features[0:7]",
                            "encoder.features[0:2]")
        skip_frozen = set()
        if num_channels != pretrained_channels:
            for layer_name in frozen_layers:
                if layer_name in first_conv_names:
                    skip_frozen.add(layer_name)
                    logger.warning(
                        "Auto-unfreezing '%s': model has %d input channels "
                        "(pretrained on %d). The adapted weights are not truly "
                        "pretrained and must be trainable.",
                        layer_name, num_channels, pretrained_channels)

        for layer_name in frozen_layers:
            if layer_name in skip_frozen:
                continue
            self._freeze_layer(model, layer_name)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created: {trainable:,}/{total:,} parameters trainable "
                   f"({100*trainable/total:.1f}%)")
        return model

    def _freeze_layer(self, model, layer_name: str):
        """Freeze a specific layer by name."""
        try:
            # Handle encoder prefix
            if layer_name.startswith("encoder."):
                parts = layer_name[8:].split(".")  # Remove "encoder."
                current = model.encoder
            elif layer_name == "decoder":
                for param in model.decoder.parameters():
                    param.requires_grad = False
                return
            elif layer_name == "segmentation_head":
                for param in model.segmentation_head.parameters():
                    param.requires_grad = False
                return
            else:
                parts = layer_name.split(".")
                current = model

            # Navigate to the layer, handling slice notation
            for part in parts:
                if "[" in part:
                    # Handle slice notation like "_blocks[0:4]" or "features[7:14]"
                    sub_modules = self._resolve_slice_modules(current, part)
                    if not sub_modules:
                        logger.warning("Could not resolve slice '%s' in layer %s",
                                       part, layer_name)
                        return
                    frozen_count = 0
                    for sub_module in sub_modules:
                        for param in sub_module.parameters():
                            param.requires_grad = False
                            frozen_count += 1
                    logger.debug("Frozen layer: %s (%d params across %d sub-modules)",
                                 layer_name, frozen_count, len(sub_modules))
                    return
                if hasattr(current, part):
                    current = getattr(current, part)

            # Freeze all parameters
            for param in current.parameters():
                param.requires_grad = False

            logger.debug("Frozen layer: %s", layer_name)

        except Exception as e:
            logger.warning("Could not freeze layer %s: %s", layer_name, e)


    # =========================================================================
    # MULTI-CHANNEL SUPPORT - PLACEHOLDER
    # =========================================================================
    # The following methods are placeholders for future multi-channel
    # fluorescence model support. Currently, the extension focuses on
    # brightfield (RGB) images with ImageNet-pretrained encoders.
    #
    # Future additions may include:
    # - MicroNet pretrained encoders (better for microscopy)
    # - Support for models trained on TissueNet or similar datasets
    # - Channel-agnostic architectures
    # =========================================================================

    def get_multichannel_encoders(self) -> List[Dict[str, Any]]:
        """
        PLACEHOLDER: Get encoders suitable for multi-channel fluorescence images.

        Currently returns the same ImageNet encoders with notes about
        channel handling. Future versions may include microscopy-specific
        pretrained encoders.
        """
        # For now, use same encoders - smp handles channel adaptation
        encoders = self.list_encoders()

        # Add notes about multi-channel handling
        for enc in encoders:
            enc["multichannel_note"] = (
                "ImageNet weights are adapted for N channels by the model. "
                "For >3 channels, encoder weights are repeated. "
                "Fine-tuning is recommended for best results."
            )

        return encoders

    def get_multichannel_recommendations(self) -> Dict[str, Any]:
        """
        PLACEHOLDER: Get recommendations for multi-channel training.
        """
        return {
            "status": "placeholder",
            "message": "Multi-channel support uses ImageNet encoders with channel adaptation",
            "recommendations": [
                "Use smp with in_channels=N for your channel count",
                "Consider per-channel normalization (percentile_99)",
                "Fine-tune all encoder layers for domain adaptation",
                "Larger datasets needed for multi-channel training"
            ],
            "future_plans": [
                "MicroNet pretrained encoders for microscopy",
                "TissueNet-based pretrained models",
                "Specialized fluorescence architectures"
            ]
        }


# Global instance
_pretrained_service: Optional[PretrainedModelsService] = None


def get_pretrained_service() -> PretrainedModelsService:
    """Get or create the pretrained models service."""
    global _pretrained_service
    if _pretrained_service is None:
        _pretrained_service = PretrainedModelsService()
    return _pretrained_service
