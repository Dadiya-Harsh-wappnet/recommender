import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

class MarbleTrainerFinder:
    def __init__(self, 
                 train_path="E:/Wappnet internship/marble_tiles/WAPPnet09/WAPPnet/BACKUP MARBLE",
                 search_path="E:/Wappnet internship/marble_tiles/WAPPnet09/WAPPnet/FIND THIS MARBLE"):
        self.train_path = Path(train_path)
        self.search_path = Path(search_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Initialize storage for training data
        self.trained_features = None
        self.trained_images = None
        self.nn_model = None
        
        # Create output directory for results
        self.output_dir = self.search_path / "similarity_results"
        self.output_dir.mkdir(exist_ok=True)
        
    @torch.no_grad()
    def extract_features(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            return features.squeeze().cpu().numpy()
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            return None
    
    def train_on_backup(self):
        """Train the model on backup marble images"""
        print("Training on backup marble images...")
        
        # Scan for training images
        image_extensions = ('.jpg', '.jpeg', '.png')
        train_files = []
        
        for ext in image_extensions:
            train_files.extend(list(self.train_path.glob(f'**/*{ext}')))
        
        if not train_files:
            raise Exception("No training images found!")
            
        print(f"Found {len(train_files)} training images")
        
        # Process training images
        features_list = []
        valid_images = []
        
        print("\nExtracting features from training images...")
        for img_path in tqdm(train_files):
            features = self.extract_features(img_path)
            if features is not None:
                features_list.append(features)
                valid_images.append({
                    'path': str(img_path),
                    'name': img_path.stem,
                    'category': img_path.parent.name
                })
        
        if not valid_images:
            raise Exception("No valid training images processed!")
        
        # Store training data
        self.trained_features = np.array(features_list)
        self.trained_images = valid_images
        
        # Initialize nearest neighbors model
        print("\nInitializing nearest neighbors model...")
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn_model.fit(self.trained_features)
        
        print(f"Training complete! Processed {len(valid_images)} images successfully.")
    
    def find_similar_marbles(self):
        """Find similar marbles for all images in search directory"""
        if self.trained_features is None:
            raise Exception("Model not trained! Run train_on_backup() first.")
        
        # Scan for search images
        image_extensions = ('.jpg', '.jpeg', '.png')
        search_files = []
        
        for ext in image_extensions:
            search_files.extend(list(self.search_path.glob(f'**/*{ext}')))
        
        if not search_files:
            raise Exception("No search images found!")
            
        print(f"\nProcessing {len(search_files)} search images...")
        
        # Process each search image
        results_dict = {}
        
        for img_path in tqdm(search_files):
            features = self.extract_features(img_path)
            if features is not None:
                # Find nearest neighbors
                distances, indices = self.nn_model.kneighbors(
                    features.reshape(1, -1),
                    n_neighbors=min(5, len(self.trained_images))
                )
                
                # Store results
                similar_images = []
                for idx, distance in zip(indices[0], distances[0]):
                    similar_images.append({
                        'path': self.trained_images[idx]['path'],
                        'name': self.trained_images[idx]['name'],
                        'category': self.trained_images[idx]['category'],
                        'similarity_score': round((1 - distance) * 100, 2)
                    })
                
                results_dict[str(img_path.relative_to(self.search_path))] = {
                    'name': img_path.stem,
                    'category': img_path.parent.name,
                    'similar_images': similar_images
                }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f'similarity_results_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4)
        
        # Generate HTML report
        html_report = self.output_dir / f'similarity_report_{timestamp}.html'
        self.generate_html_report(results_dict, html_report)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to: {results_file}")
        print(f"HTML report saved to: {html_report}")
    
    def generate_html_report(self, results_dict, output_file):
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Marble Similarity Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .marble-entry { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .similar-marble { margin: 10px; padding: 10px; background-color: #f9f9f9; }
                .similarity-score { color: #007bff; font-weight: bold; }
                .category { color: #28a745; }
                .search-image { font-weight: bold; color: #dc3545; }
            </style>
        </head>
        <body>
            <h1>Marble Similarity Report</h1>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        """
        
        for search_path, data in results_dict.items():
            html_content += f"""
            <div class="marble-entry">
                <h2>Search Image: <span class="search-image">{data['name']}</span></h2>
                <p>Path: {search_path}</p>
                <p>Category: <span class="category">{data['category']}</span></p>
                <h3>Similar Marbles from Backup:</h3>
            """
            
            for similar in data['similar_images']:
                html_content += f"""
                <div class="similar-marble">
                    <p>Name: {similar['name']}</p>
                    <p>Path: {similar['path']}</p>
                    <p>Category: <span class="category">{similar['category']}</span></p>
                    <p>Similarity Score: <span class="similarity-score">{similar['similarity_score']}%</span></p>
                </div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    # Initialize the processor
    processor = MarbleTrainerFinder()
    
    # Train on backup images
    processor.train_on_backup()
    
    # Find similar marbles
    processor.find_similar_marbles()

if __name__ == "__main__":
    main()