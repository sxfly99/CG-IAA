import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

from models.caption_model import AesCritique


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate multi-attribute aesthetic captions for a single image"
    )
    
    parser.add_argument(
        '--image_path', 
        type=str, 
        required=True,
        help='Path to the input image'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    
    # Model weight paths (optional, defaults to checkpoints/)
    parser.add_argument(
        '--clip_model_path',
        type=str,
        default='checkpoints/base_model.pt',
        help='Path to CLIP base model weights'
    )
    
    parser.add_argument(
        '--color_model_path',
        type=str,
        default='checkpoints/color.pt',
        help='Path to color expert model weights'
    )
    
    parser.add_argument(
        '--composition_model_path',
        type=str,
        default='checkpoints/composition.pt',
        help='Path to composition expert model weights'
    )
    
    parser.add_argument(
        '--dof_model_path',
        type=str,
        default='checkpoints/dof.pt',
        help='Path to depth of field expert model weights'
    )
    
    parser.add_argument(
        '--general_model_path',
        type=str,
        default='checkpoints/general.pt',
        help='Path to general expert model weights'
    )
    
    args = parser.parse_args()
    return args


def load_image(image_path):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to the input image
    
    Returns:
        Preprocessed image tensor
    """
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def generate_caption(
    image_path, 
    device='cuda',
    clip_model_path='checkpoints/base_model.pt',
    color_model_path='checkpoints/color.pt',
    composition_model_path='checkpoints/composition.pt',
    dof_model_path='checkpoints/dof.pt',
    general_model_path='checkpoints/general.pt'
):
    """
    Generate multi-attribute aesthetic captions for a single image
    
    Args:
        image_path: Path to the input image
        device: Device to run inference on ('cuda' or 'cpu')
        clip_model_path: Path to CLIP base model (default: 'checkpoints/base_model.pt')
        color_model_path: Path to color expert model (default: 'checkpoints/color.pt')
        composition_model_path: Path to composition expert model (default: 'checkpoints/composition.pt')
        dof_model_path: Path to depth of field expert model (default: 'checkpoints/dof.pt')
        general_model_path: Path to general expert model (default: 'checkpoints/general.pt')
    
    Returns:
        Dictionary containing captions for each attribute
    """
    # Load model with custom paths if provided
    model = AesCritique(
        clip_model_path=clip_model_path,
        color_model_path=color_model_path,
        composition_model_path=composition_model_path,
        dof_model_path=dof_model_path,
        general_model_path=general_model_path
    )
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Load and preprocess image
    image_tensor = load_image(image_path).to(device)
    
    # Generate captions
    with torch.no_grad():
        color_comment, composition_comment, dof_comment, general_comment = model(image_tensor)
    
    # Organize results
    results = {
        'image_path': image_path,
        'color': color_comment[0],
        'composition': composition_comment[0],
        'depth_of_field': dof_comment[0],
        'general': general_comment[0]
    }
    
    return results


def main():
    """Main function"""
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    
    print(f"Loading model and processing image: {args.image_path}")
    
    # Generate captions with custom model paths if provided
    results = generate_caption(
        args.image_path, 
        args.device,
        clip_model_path=args.clip_model_path,
        color_model_path=args.color_model_path,
        composition_model_path=args.composition_model_path,
        dof_model_path=args.dof_model_path,
        general_model_path=args.general_model_path
    )
    
    # Print results
    print("\n" + "="*80)
    print(f"Multi-Attribute Aesthetic Critiques for: {results['image_path']}")
    print("="*80)
    print(f"\n[Color]\n{results['color']}")
    print(f"\n[Composition]\n{results['composition']}")
    print(f"\n[Depth of Field]\n{results['depth_of_field']}")
    print(f"\n[General]\n{results['general']}")
    print("\n" + "="*80 + "\n")
    

if __name__ == "__main__":
    main()
