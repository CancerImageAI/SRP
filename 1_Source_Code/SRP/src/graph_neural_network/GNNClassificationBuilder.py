import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import gc


def set_gpu(gpu_id):
    """Set specific GPU device"""
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Setting GPU device to: {gpu_id}")
    else:
        print("Using CPU")


class ResponseGraphDataset(Dataset):
    """
    Dataset for Response prediction using voxel graphs
    """
    
    def __init__(self, 
                 graph_dir: str,
                 clinical_df: pd.DataFrame,
                 case_id_col: str = 'CaseID',
                 label_col: str = 'Response',
                 use_clinical_features: bool = True,
                 clinical_feature_cols: List[str] = None,
                 scaler: Optional[StandardScaler] = None,
                 graph_file_extension: str = '.gpickle',
                 max_nodes: Optional[int] = None):
        """
        Args:
            graph_dir: Directory containing graph files
            clinical_df: DataFrame with clinical/radiomics features and labels
            case_id_col: Column name for case IDs
            label_col: Column name for Response labels
            use_clinical_features: Whether to include clinical features
            clinical_feature_cols: List of clinical feature column names
            scaler: Pre-fitted scaler for clinical features (for test set)
            graph_file_extension: Extension of graph files
            max_nodes: Maximum number of nodes to keep per graph
        """
        super().__init__()
        
        self.graph_dir = Path(graph_dir)
        self.clinical_df = clinical_df
        self.case_id_col = case_id_col
        self.label_col = label_col
        self.use_clinical_features = use_clinical_features
        self.clinical_feature_cols = clinical_feature_cols
        self.graph_file_extension = graph_file_extension
        self.max_nodes = max_nodes
        
        # Check if directory exists
        if not self.graph_dir.exists():
            raise ValueError(f"Graph directory does not exist: {self.graph_dir}")
        
        # Get list of available graph files and their case IDs
        self.graph_files = []
        self.case_ids = []
        
        # Convert clinical case IDs to set for faster lookup
        clinical_case_ids = set(clinical_df[case_id_col].astype(str).values)
        
        print(f"\n{'='*70}")
        print(f"Loading Graph Files")
        print(f"{'='*70}")
        print(f"Graph directory: {self.graph_dir}")
        print(f"Looking for files with extension: {graph_file_extension}")
        
        # Search for graph files
        if graph_file_extension.startswith('.'):
            pattern = f'*{graph_file_extension}'
        else:
            pattern = f'*.{graph_file_extension}'
        
        all_graph_files = list(self.graph_dir.glob(pattern))
        print(f"Found {len(all_graph_files)} {graph_file_extension} files")
        
        matched_count = 0
        
        for graph_file in all_graph_files:
            filename = graph_file.stem
            # Extract case ID (handle different naming formats)
            case_id = filename.replace('_graph', '').split('_')[0]
            
            if case_id in clinical_case_ids:
                self.graph_files.append(graph_file)
                self.case_ids.append(case_id)
                matched_count += 1
        
        print(f"Matched cases: {matched_count}")
        
        if len(self.graph_files) == 0:
            raise ValueError(f"No matching graphs found!")
        
        # Process labels
        self._process_labels()
        
        # Get clinical features if requested
        if self.use_clinical_features:
            self._prepare_clinical_features(scaler)
    
    def _process_labels(self):
        """Process Response labels to binary format"""
        self.labels = []
        
        for case_id in self.case_ids:
            label = self.clinical_df[self.clinical_df[self.case_id_col].astype(str) == case_id][self.label_col].values[0]
            if isinstance(label, str):
                binary_label = 1 if label == 'pCR' else 0
            else:
                binary_label = int(label > 0)
            
            self.labels.append(binary_label)
        self.labels = np.array(self.labels)
        
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\nLabel distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(self.labels)*100:.1f}%)")
    
    def _prepare_clinical_features(self, scaler=None):
        """Prepare clinical/radiomics features"""
        if self.clinical_feature_cols is None:
            exclude = [self.case_id_col, self.label_col, 'Response_binary']
            diagnostic_cols = [col for col in self.clinical_df.columns 
                             if 'diagnostics' in col.lower()]
            exclude.extend(diagnostic_cols)
            
            numeric_cols = self.clinical_df.select_dtypes(include=[np.number]).columns.tolist()
            self.clinical_feature_cols = [col for col in numeric_cols if col not in exclude]
        
        if len(self.clinical_feature_cols) == 0:
            print("\nNo clinical features found - using graph features only.")
            self.use_clinical_features = False
            return
        
        print(f"\nUsing {len(self.clinical_feature_cols)} clinical/radiomics features")
        
        self.clinical_features = []
        for case_id in self.case_ids:
            features = self.clinical_df[
                self.clinical_df[self.case_id_col].astype(str) == case_id
            ][self.clinical_feature_cols].values[0]
            self.clinical_features.append(features)
        
        self.clinical_features = np.array(self.clinical_features, dtype=np.float32)
        
        if np.isnan(self.clinical_features).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.clinical_features = imputer.fit_transform(self.clinical_features)
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.clinical_features = self.scaler.fit_transform(self.clinical_features)
        else:
            self.scaler = scaler
            self.clinical_features = self.scaler.transform(self.clinical_features)
    
    def len(self):
        return len(self.graph_files)
    
    def get(self, idx):
        """
        Load graph and prepare PyG Data object
        """
        graph_path = self.graph_files[idx]
        
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        # Get nodes (potentially subsample if graph is too large)
        nodes = list(G.nodes())
        
        if self.max_nodes is not None and len(nodes) > self.max_nodes:
            # Sample nodes by degree (keep most connected nodes)
            degrees = dict(G.degree())
            sorted_nodes = sorted(nodes, key=lambda x: degrees[x], reverse=True)
            nodes = sorted_nodes[:self.max_nodes]
            G = G.subgraph(nodes).copy()
        
        # Convert to PyG format
        node_features = []
        node_to_idx = {}
        
        for node_idx, node in enumerate(G.nodes()):
            node_to_idx[node] = node_idx
            embedding = G.nodes[node]['embedding']
            
            # Optionally add spatial features
            if 'spatial_features' in G.nodes[node]:
                spatial = G.nodes[node]['spatial_features']
                features = np.concatenate([embedding, spatial])
            else:
                features = embedding
            
            node_features.append(features)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Edge index and edge attributes
        edge_index = []
        edge_attr = []
        
        for edge in G.edges():
            i, j = edge
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                
                edge_index.append([idx_i, idx_j])
                edge_index.append([idx_j, idx_i])
                
                weight = G[i][j]['weight']
                edge_attr.append([weight])
                edge_attr.append([weight])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        y = torch.tensor([self.labels[idx]], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        if self.use_clinical_features:
            data.clinical_features = torch.tensor(self.clinical_features[idx], dtype=torch.float)
        
        data.case_id = self.case_ids[idx]
        
        # Clean up
        del G
        
        return data


class ResponseGNN(nn.Module):
    """
    Graph Neural Network for Response Expression Prediction
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 gnn_type: str = 'GAT',
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 clinical_feature_dim: int = 0,
                 pooling: str = 'mean',
                 use_gradient_checkpointing: bool = False):
        """
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            num_gnn_layers: Number of GNN layers
            gnn_type: Type of GNN ('GCN', 'GAT', 'SAGE')
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            clinical_feature_dim: Dimension of clinical features
            pooling: Graph pooling method ('mean', 'max', 'both')
            use_gradient_checkpointing: Use gradient checkpointing to save memory
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.clinical_feature_dim = clinical_feature_dim
        self.pooling = pooling
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_gnn_layers):
            if gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )
            elif gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Determine graph-level feature dimension
        if pooling == 'both':
            graph_feature_dim = hidden_dim * 2
        else:
            graph_feature_dim = hidden_dim
        
        # Clinical feature integration
        if clinical_feature_dim > 0:
            self.clinical_proj = nn.Sequential(
                nn.Linear(clinical_feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            combined_dim = graph_feature_dim + hidden_dim
        else:
            combined_dim = graph_feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
            
    def forward(self, data):
        """
        Forward pass
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers with residual connections
        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._gnn_forward, x, edge_index, gnn_layer, bn, i
                )
            else:
                x = self._gnn_forward(x, edge_index, gnn_layer, bn, i)
        
        # Graph-level pooling
        if self.pooling == 'mean':
            graph_features = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            graph_features = global_max_pool(x, batch)
        elif self.pooling == 'both':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            graph_features = torch.cat([mean_pool, max_pool], dim=1)
        
        # Integrate clinical features if available
        if self.clinical_feature_dim > 0 and hasattr(data, 'clinical_features'):
            clinical_feat = self.clinical_proj(data.clinical_features)
            combined_features = torch.cat([graph_features, clinical_feat], dim=1)
        else:
            combined_features = graph_features
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def _gnn_forward(self, x, edge_index, gnn_layer, bn, layer_idx):
        """Helper function for gradient checkpointing"""
        x_residual = x
        x = gnn_layer(x, edge_index)
        x = bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Residual connection (skip first layer)
        if layer_idx > 0:
            x = x + x_residual
        
        return x


class ResponseTrainer:
    """
    Trainer for Response prediction model
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 class_weights: Optional[torch.Tensor] = None,
                 mixed_precision: bool = True):
        """
        Args:
            model: ResponseGNN model
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            class_weights: Weights for handling class imbalance
            mixed_precision: Use mixed precision training (saves memory)
        """
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Mixed precision scaler
        if mixed_precision and device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training (FP16)")
        else:
            self.scaler = None
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_accs = []
        
        # NEW: Store best model state
        self.best_model_state = None
        self.best_val_auc = 0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for data in tqdm(train_loader, desc="Training"):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(data)
                    loss = self.criterion(logits, data.y)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                logits = self.model(data)
                loss = self.criterion(logits, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            
            # Store predictions
            preds = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(data.y.cpu().numpy())
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        
        return avg_loss, auc, acc

    def evaluate(self, val_loader, return_detailed=False):
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_case_ids = []
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Evaluating"):
                data = data.to(self.device)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(data)
                        loss = self.criterion(logits, data.y)
                else:
                    logits = self.model(data)
                    loss = self.criterion(logits, data.y)
                
                total_loss += loss.item() * data.num_graphs
                
                preds = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(data.y.cpu().numpy())
                all_case_ids.extend(data.case_id)
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(val_loader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, (all_preds > 0.5).astype(int), average='binary'
        )
        
        cm = confusion_matrix(all_labels, (all_preds > 0.5).astype(int))
        
        results = {
            'loss': avg_loss,
            'auc': auc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'case_ids': all_case_ids
        }
        
        return results

    def train(self, train_loader, val_loader, num_epochs=100, early_stopping_patience=20, output_dir=Path('./results')):
        """Complete training loop with best model tracking"""
        patience_counter = 0
        
        print(f"\n{'='*70}")
        print("Starting Training")
        print(f"{'='*70}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_auc, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_aucs.append(val_metrics['auc'])
            self.val_accs.append(val_metrics['accuracy'])
            
            # Update learning rate
            self.scheduler.step(val_metrics['auc'])
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Memory info
            if torch.cuda.is_available():
                print(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            
            # NEW: Track best model based on validation AUC
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_epoch = epoch + 1
                patience_counter = 0
                # Save best model state (deep copy)
                self.best_model_state = {
                    'model_state_dict': self.model.state_dict().copy(),
                    'optimizer_state_dict': self.optimizer.state_dict().copy(),
                    'epoch': epoch + 1,
                    'val_auc': val_metrics['auc'],
                    'val_acc': val_metrics['accuracy'],
                    'val_loss': val_metrics['loss']
                }
                # Save best model
                model_path = output_dir / 'best_model.pt'
                self.save_model(str(model_path))
                print(f"  ✓ New best model (AUC: {self.best_val_auc:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            print(f"\n{'='*70}")
            print(f"Loaded BEST model from epoch {self.best_epoch}")
            print(f"  Best Validation AUC: {self.best_val_auc:.4f}")
            print(f"  Best Validation Acc: {self.best_model_state['val_acc']:.4f}")
            print(f"{'='*70}")
        
        return self.best_val_auc

    def save_model(self, path: str):
        """Save best model checkpoint"""
        if self.best_model_state is None:
            print("Warning: No best model state found, saving current model")
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_aucs': self.val_aucs,
                'val_accs': self.val_accs,
                'best_val_auc': self.best_val_auc,
                'best_epoch': self.best_epoch
            }
        else:
            checkpoint = {
                'model_state_dict': self.best_model_state['model_state_dict'],
                'optimizer_state_dict': self.best_model_state['optimizer_state_dict'],
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_aucs': self.val_aucs,
                'val_accs': self.val_accs,
                'best_val_auc': self.best_val_auc,
                'best_epoch': self.best_epoch,
                'best_model_metrics': {
                    'val_auc': self.best_model_state['val_auc'],
                    'val_acc': self.best_model_state['val_acc'],
                    'val_loss': self.best_model_state['val_loss']
                }
            }
        
        torch.save(checkpoint, path)
        print(f"Best model saved to: {path}")
        print(f"  From epoch: {self.best_epoch}")
        print(f"  Best AUC: {self.best_val_auc:.4f}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_aucs = checkpoint.get('val_aucs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        self.best_val_auc = checkpoint.get('best_val_auc', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        print(f"Model loaded from: {path}")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best AUC: {self.best_val_auc:.4f}")

    def plot_training_curves(self, output_path: str):
        """Plot training curves with best epoch marker"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Validation', linewidth=2)
        if self.best_epoch > 0:
            axes[0, 0].axvline(x=self.best_epoch-1, color='r', linestyle='--', 
                              linewidth=2, label=f'Best (Epoch {self.best_epoch})')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[0, 1].plot(self.val_aucs, label='Validation AUC', color='green', linewidth=2)
        if self.best_epoch > 0:
            axes[0, 1].axvline(x=self.best_epoch-1, color='r', linestyle='--', 
                              linewidth=2, label=f'Best (Epoch {self.best_epoch})')
            axes[0, 1].scatter([self.best_epoch-1], [self.best_val_auc], 
                              color='red', s=100, zorder=5, label=f'Best AUC: {self.best_val_auc:.4f}')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('AUC', fontsize=12)
        axes[0, 1].set_title('Validation AUC', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.4, 1.0])
        # Accuracy
        axes[1, 0].plot(self.val_accs, label='Validation Accuracy', color='orange', linewidth=2)
        if self.best_epoch > 0:
            axes[1, 0].axvline(x=self.best_epoch-1, color='r', linestyle='--', 
                              linewidth=2, label=f'Best (Epoch {self.best_epoch})')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy', fontsize=12)
        axes[1, 0].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0.4, 1.0])
        
        # Best metrics summary
        summary_text = f"Best Model (Epoch {self.best_epoch}):\n\n"
        summary_text += f"Best Val AUC: {self.best_val_auc:.4f}\n"
        if self.best_epoch > 0 and self.best_epoch <= len(self.val_accs):
            summary_text += f"Best Val Acc: {self.val_accs[self.best_epoch-1]:.4f}\n"
        summary_text += f"\nFinal Epoch ({len(self.val_aucs)}):\n"
        summary_text += f"AUC: {self.val_aucs[-1]:.4f}\n"
        summary_text += f"Acc: {self.val_accs[-1]:.4f}\n"
        summary_text += f"\nTotal Epochs: {len(self.train_losses)}"
        
        axes[1, 1].text(0.5, 0.5, summary_text, 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {output_path}")


def plot_test_results(test_metrics: dict, output_dir: str, title_prefix: str = "Test Set"):
    """Plot comprehensive test results"""
    output_dir = Path(output_dir)
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cm = test_metrics['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {title_prefix}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{title_prefix.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    fpr, tpr, thresholds = roc_curve(test_metrics['labels'], test_metrics['predictions'])
    
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {test_metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {title_prefix}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{title_prefix.lower().replace(" ", "_")}_roc_curve.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Prediction Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    predictions = test_metrics['predictions']
    labels = test_metrics['labels']
    
    axes[0].hist(predictions[labels == 0], bins=30, alpha=0.6, label='Negative', color='blue')
    axes[0].hist(predictions[labels == 1], bins=30, alpha=0.6, label='Positive', color='red')
    axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Prediction Distribution by True Label - {title_prefix}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    data_to_plot = [predictions[labels == 0], predictions[labels == 1]]
    axes[1].boxplot(data_to_plot, labels=['Negative', 'Positive'])
    axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1].set_ylabel('Predicted Probability', fontsize=12)
    axes[1].set_xlabel('True Label', fontsize=12)
    axes[1].set_title(f'Prediction Distribution (Box Plot) - {title_prefix}', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{title_prefix.lower().replace(" ", "_")}_prediction_distribution.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Metrics Summary
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [
        test_metrics['auc'],
        test_metrics['accuracy'],
        test_metrics['precision'],
        test_metrics['recall'],
        test_metrics['f1']
    ]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.barh(metrics_names, metrics_values, color=colors, alpha=0.8)
    
    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        ax.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim([0, 1.1])
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title(f'{title_prefix} Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{title_prefix.lower().replace(" ", "_")}_metrics_summary.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"{title_prefix} result plots saved to: {output_dir}")


def save_predictions_to_csv(metrics: dict, output_path: str, split_name: str):
    """Save predictions to CSV file"""
    predictions_df = pd.DataFrame({
        'case_id': metrics['case_ids'],
        'true_label': metrics['labels'],
        'predicted_prob': metrics['predictions'],
        'predicted_label': (metrics['predictions'] > 0.5).astype(int),
        'split': split_name
    })
    
    predictions_df.to_csv(output_path, index=False)
    print(f"{split_name} predictions saved to: {output_path}")
    
    return predictions_df


def train_test_split_experiment(dataset: ResponseGraphDataset,
                                model_config: dict,
                                test_size: float = 0.3,
                                batch_size: int = 8,
                                num_epochs: int = 100,
                                output_dir: str = './results',
                                random_state: int = 42):
    """
    Train model with 70:30 train-test split
    Returns predictions for both training and test sets using BEST model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = np.array([data.y.item() for data in dataset])
    indices = np.arange(len(labels))
    
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    print(f"\n{'='*70}")
    print(f"Dataset Split (70% Train / 30% Test)")
    print(f"{'='*70}")
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_idx)} ({len(train_idx)/len(dataset)*100:.1f}%)")
    print(f"Test samples: {len(test_idx)} ({len(test_idx)/len(dataset)*100:.1f}%)")
    
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    print(f"\nTraining set label distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} ({count/len(train_labels)*100:.1f}%)")
    
    print(f"\nTest set label distribution:")
    unique, counts = np.unique(test_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} ({count/len(test_labels)*100:.1f}%)")
    
    # Create data loaders

    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=32,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=32,
        pin_memory=True
    )
    
    # Calculate class weights
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor(len(train_labels) / (len(class_counts) * class_counts))
    
    print(f"\nClass weights for training: {class_weights.numpy()}")
    
    # Initialize model
    print(f"\n{'='*70}")
    print("Initializing Model")
    print(f"{'='*70}")
    print(f"Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    model = ResponseGNN(**model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize trainer
    trainer = ResponseTrainer(
        model=model,
        device=device,
        learning_rate=1e-5,
        weight_decay=1e-5,
        class_weights=class_weights,
        mixed_precision=True
    )
    
    # Train
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}")
    
    best_val_auc = trainer.train(
        train_loader, 
        test_loader, 
        num_epochs=num_epochs,
        early_stopping_patience=40,
        output_dir=output_dir
    )
    model_path = output_dir / 'best_model.pt'
    trainer.load_model(str(model_path))
    # ========================================================================
    # NEW: Evaluate BEST model on BOTH training and test sets
    # ========================================================================
    print(f"\n{'='*70}")
    print("Evaluating BEST Model on Training and Test Sets")
    print(f"{'='*70}")
    
    # Evaluate on training set
    print("\nEvaluating on Training Set...")
    train_metrics = trainer.evaluate(train_loader)
    
    print(f"\nTraining Set Results (Best Model from Epoch {trainer.best_epoch}):")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  AUC: {train_metrics['auc']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall: {train_metrics['recall']:.4f}")
    print(f"  F1-Score: {train_metrics['f1']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on Test Set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nTest Set Results (Best Model from Epoch {trainer.best_epoch}):")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    print(test_metrics['confusion_matrix'])
    
    
    # ========================================================================
    # NEW: Save predictions for BOTH training and test sets
    # ========================================================================
    print(f"\n{'='*70}")
    print("Saving Predictions")
    print(f"{'='*70}")
    # Save training set predictions
    train_pred_path = output_dir / 'train_predictions.csv'
    train_pred_df = save_predictions_to_csv(train_metrics, str(train_pred_path), 'train')
    
    # Save test set predictions
    test_pred_path = output_dir / 'test_predictions.csv'
    test_pred_df = save_predictions_to_csv(test_metrics, str(test_pred_path), 'test')
    
    # Combine and save both
    combined_pred_df = pd.concat([train_pred_df, test_pred_df], ignore_index=True)
    combined_pred_path = output_dir / 'all_predictions.csv'
    combined_pred_df.to_csv(combined_pred_path, index=False)
    print(f"Combined predictions saved to: {combined_pred_path}")
    
    # ========================================================================
    # Generate plots for both sets
    # ========================================================================
    print(f"\n{'='*70}")
    print("Generating Plots")
    print(f"{'='*70}")
    
    # Training curves
    curves_path = output_dir / 'training_curves.png'
    trainer.plot_training_curves(str(curves_path))
    
    # Training set plots
    print("\nGenerating training set plots...")
    plot_test_results(train_metrics, output_dir, title_prefix="Training Set")
    
    # Test set plots
    print("\nGenerating test set plots...")
    plot_test_results(test_metrics, output_dir, title_prefix="Test Set")
    
    # ========================================================================
    # Create comparison plot
    # ========================================================================
    print("\nGenerating comparison plots...")
    create_train_test_comparison_plots(train_metrics, test_metrics, output_dir)
    
    # ========================================================================
    # Save comprehensive metrics summary
    # ========================================================================
    metrics_summary = {
        'best_model_info': {
            'best_epoch': trainer.best_epoch,
            'best_val_auc': trainer.best_val_auc,
            'total_epochs_trained': len(trainer.train_losses)
        },
        'train_metrics': {
            'loss': float(train_metrics['loss']),
            'auc': float(train_metrics['auc']),
            'accuracy': float(train_metrics['accuracy']),
            'precision': float(train_metrics['precision']),
            'recall': float(train_metrics['recall']),
            'f1': float(train_metrics['f1']),
            'confusion_matrix': train_metrics['confusion_matrix'].tolist()
        },
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'auc': float(test_metrics['auc']),
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
        },
        'dataset_info': {
            'total_samples': len(dataset),
            'train_samples': len(train_idx),
            'test_samples': len(test_idx),
            'train_split': 1 - test_size,
            'test_split': test_size,
            'train_label_distribution': {
                int(k): int(v) for k, v in zip(*np.unique(train_labels, return_counts=True))
            },
            'test_label_distribution': {
                int(k): int(v) for k, v in zip(*np.unique(test_labels, return_counts=True))
            }
        },
        'model_config': model_config,
        'training_config': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 1e-5,
            'weight_decay': 1e-5,
            'early_stopping_patience': 20,
            'class_weights': class_weights.tolist(),
            'mixed_precision': True
        }
    }
    
    summary_path = output_dir / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Experiment summary saved to: {summary_path}")
    
    # ========================================================================
    # Print final summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE - SUMMARY")
    print(f"{'='*70}")
    print(f"\nBest Model: Epoch {trainer.best_epoch} (out of {len(trainer.train_losses)})")
    print(f"\nTraining Set Performance:")
    print(f"  AUC: {train_metrics['auc']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall: {train_metrics['recall']:.4f}")
    print(f"  F1-Score: {train_metrics['f1']:.4f}")
    
    print(f"\nTest Set Performance:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return train_metrics, test_metrics, trainer


def create_train_test_comparison_plots(train_metrics: dict, test_metrics: dict, output_dir: str):
    """
    Create comparison plots between training and test sets
    """
    output_dir = Path(output_dir)
    
    # 1. Metrics comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [
        train_metrics['auc'],
        train_metrics['accuracy'],
        train_metrics['precision'],
        train_metrics['recall'],
        train_metrics['f1']
    ]
    test_values = [
        test_metrics['auc'],
        test_metrics['accuracy'],
        test_metrics['precision'],
        test_metrics['recall'],
        test_metrics['f1']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax = axes[0]
    bars1 = ax.bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Training vs Test Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # 2. ROC curves comparison
    ax = axes[1]
    
    # Training ROC
    fpr_train, tpr_train, _ = roc_curve(train_metrics['labels'], train_metrics['predictions'])
    ax.plot(fpr_train, tpr_train, linewidth=2, 
            label=f"Training (AUC={train_metrics['auc']:.3f})", color='#3498db')
    
    # Test ROC
    fpr_test, tpr_test, _ = roc_curve(test_metrics['labels'], test_metrics['predictions'])
    ax.plot(fpr_test, tpr_test, linewidth=2, 
            label=f"Test (AUC={test_metrics['auc']:.3f})", color='#e74c3c')
    
    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'train_test_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training confusion matrix
    ax = axes[0]
    sns.heatmap(train_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Training Set Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Test confusion matrix
    ax = axes[1]
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Test Set Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training set distributions
    train_preds = train_metrics['predictions']
    train_labels = train_metrics['labels']
    
    axes[0, 0].hist(train_preds[train_labels == 0], bins=30, alpha=0.6, label='Negative', color='blue')
    axes[0, 0].hist(train_preds[train_labels == 1], bins=30, alpha=0.6, label='Positive', color='red')
    axes[0, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Training Set - Prediction Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test set distributions
    test_preds = test_metrics['predictions']
    test_labels = test_metrics['labels']
    
    axes[0, 1].hist(test_preds[test_labels == 0], bins=30, alpha=0.6, label='Negative', color='blue')
    axes[0, 1].hist(test_preds[test_labels == 1], bins=30, alpha=0.6, label='Positive', color='red')
    axes[0, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Probability', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Test Set - Prediction Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training box plots
    train_data_to_plot = [train_preds[train_labels == 0], train_preds[train_labels == 1]]
    bp1 = axes[1, 0].boxplot(train_data_to_plot, labels=['Negative', 'Positive'],
                              patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    axes[1, 0].axhline(y=0.5, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_ylabel('Predicted Probability', fontsize=12)
    axes[1, 0].set_xlabel('True Label', fontsize=12)
    axes[1, 0].set_title('Training Set - Box Plot', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Test box plots
    test_data_to_plot = [test_preds[test_labels == 0], test_preds[test_labels == 1]]
    bp2 = axes[1, 1].boxplot(test_data_to_plot, labels=['Negative', 'Positive'],
                              patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    axes[1, 1].axhline(y=0.5, color='black', linestyle='--', linewidth=2)
    axes[1, 1].set_ylabel('Predicted Probability', fontsize=12)
    axes[1, 1].set_xlabel('True Label', fontsize=12)
    axes[1, 1].set_title('Test Set - Box Plot', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distributions_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {output_dir}")


def main(args):
    """
    Main execution function
    """
    print(f"\n{'='*70}")
    print("Loading Clinical Data")
    print(f"{'='*70}")
    
    clinical_df = pd.read_excel(args.clinical_xlsx)
    
    # Process labels
    clinical_df['Response_binary'] = clinical_df['Response'].apply(
        lambda x: 0 if (isinstance(x, str) and x == 'pCR') or x == 0 else 1
    )
    
    print(f"Loaded {len(clinical_df)} cases from: {args.clinical_xlsx}")
    print(f"\nLabel distribution:")
    print(clinical_df['Response_binary'].value_counts())
    
    # Prepare clinical features
    exclude_cols = ['CaseID', 'Response', 'Response_binary']
    diagnostic_cols = [col for col in clinical_df.columns if 'diagnostics' in col.lower()]
    exclude_cols.extend(diagnostic_cols)
    
    numeric_cols = clinical_df.select_dtypes(include=[np.number]).columns.tolist()
    clinical_feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create dataset
    print(f"\n{'='*70}")
    print("Creating Dataset")
    print(f"{'='*70}")
    
    dataset = ResponseGraphDataset(
        graph_dir=args.graph_dir,
        clinical_df=clinical_df,
        case_id_col='CaseID',
        label_col='Response_binary',
        use_clinical_features=args.use_clinical,
        clinical_feature_cols=clinical_feature_cols if args.use_clinical else None,
        graph_file_extension='.gpickle',
        max_nodes=args.max_nodes
    )
    
    # Get dimensions
    sample_data = dataset[0]
    node_feature_dim = sample_data.x.shape[1]
    clinical_dim = sample_data.clinical_features.shape[0] if hasattr(sample_data, 'clinical_features') else 0
    
    print(f"\n{'='*70}")
    print("Feature Dimensions")
    print(f"{'='*70}")
    print(f"  Node features: {node_feature_dim}")
    print(f"  Clinical features: {clinical_dim}")
    print(f"  Graph nodes (sample): {sample_data.x.shape[0]}")
    print(f"  Graph edges (sample): {sample_data.edge_index.shape[1]}")
    
    # Model configuration
    model_config = {
        'node_feature_dim': node_feature_dim,
        'hidden_dim': args.hidden_dim,
        'num_gnn_layers': args.num_layers,
        'gnn_type': args.gnn_type,
        'num_heads': args.num_heads,
        'dropout': 0.0,
        'clinical_feature_dim': clinical_dim,
        'pooling': args.pooling,
        'use_gradient_checkpointing': False
    }
    
    # Run experiment
    train_metrics, test_metrics, trainer = train_test_split_experiment(
        dataset=dataset,
        model_config=model_config,
        test_size=0.3,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        random_state=42
    )
    
    return train_metrics, test_metrics, trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Response GNN with 70:30 train-test split (Best Model Evaluation)'
    )
    
    # Data paths
    parser.add_argument('--graph_dir', type=str, required=True,
                        help='Directory containing graph files')
    parser.add_argument('--clinical_xlsx', type=str, required=True,
                        help='Path to clinical/radiomics Excel file')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    
    # Model configuration
    parser.add_argument('--gnn_type', type=str, default='GAT',
                        choices=['GCN', 'GAT', 'SAGE'],
                        help='Type of GNN layer (default: GAT)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size (default: 128)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads for GAT (default: 4)')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'both'],
                        help='Graph pooling method (default: mean)')
    parser.add_argument('--use_clinical', action='store_true',
                        help='Use clinical/radiomics features')
    
    # Training configuration
    parser.add_argument('--max_nodes', type=int, default=None,
                        help='Maximum nodes per graph (None = no limit)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs (default: 100)')
    
    args = parser.parse_args()
    
    # Set GPU
    set_gpu(args.gpu)
    
    # Print configuration
    print(f"\n{'='*70}")
    print("Configuration Summary")
    print(f"{'='*70}")
    print(f"Graph directory: {args.graph_dir}")
    print(f"Clinical data: {args.clinical_xlsx}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU ID: {args.gpu}")
    print(f"\nModel Configuration:")
    print(f"  GNN type: {args.gnn_type}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Use clinical features: {args.use_clinical}")
    print(f"\nTraining Configuration:")
    print(f"  Max nodes per graph: {args.max_nodes}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    
    # Run main
    try:
        train_metrics, test_metrics, trainer = main(args)
        print("\n✓ Training completed successfully!")
        
        # Print final summary
        print(f"\n{'='*70}")
        print("FINAL RESULTS SUMMARY (BEST MODEL)")
        print(f"{'='*70}")
        print(f"\nBest Model: Epoch {trainer.best_epoch}")
        print(f"Best Validation AUC: {trainer.best_val_auc:.4f}")
        
        print(f"\nTraining Set Performance:")
        print(f"  AUC: {train_metrics['auc']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  F1-Score: {train_metrics['f1']:.4f}")
        
        print(f"\nTest Set Performance:")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        
        print(f"\n{'='*70}")
        print(f"All results saved to: {args.output_dir}")
        print(f"{'='*70}")
        print(f"\nGenerated files:")
        print(f"  - best_model.pt (Best model from epoch {trainer.best_epoch})")
        print(f"  - train_predictions.csv (Training set predictions)")
        print(f"  - test_predictions.csv (Test set predictions)")
        print(f"  - all_predictions.csv (Combined predictions)")
        print(f"  - experiment_summary.json (Detailed metrics)")
        print(f"  - training_curves.png (Training history)")
        print(f"  - train_test_comparison.png (Performance comparison)")
        print(f"  - Various metric plots for both sets")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n✗ GPU Out of Memory Error!")
            print("\nSuggestions to fix:")
            print(f"  1. Reduce --batch_size (current: {args.batch_size})")
            print(f"  2. Reduce --max_nodes (current: {args.max_nodes})")
            print(f"  3. Reduce --hidden_dim (current: {args.hidden_dim})")
            print(f"  4. Use --gnn_type GCN instead of GAT")
            print(f"  5. Reduce --num_layers (current: {args.num_layers})")
            
            if torch.cuda.is_available():
                print(f"\nCurrent GPU memory:")
                print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"  Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                print(f"  Total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
            raise e
        else:
            raise e


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
1. Graph features only (no clinical features) - BEST MODEL SAVED
CUDA_VISIBLE_DEVICES=2 python GNNClassificationBuilder.py \
    --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
    --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
    --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_graph_only \
    --gpu 2 \
    --gnn_type SAGE \
    --hidden_dim 64 \
    --num_layers 3 \
    --num_heads 4 \
    --pooling both \
    --batch_size 512 \
    --epochs 200

# Output files will include:  SAGE
# - best_model.pt (from best validation epoch, not final epoch)
# - train_predictions.csv (predictions on training set using best model)
# - test_predictions.csv (predictions on test set using best model)
# - all_predictions.csv (combined predictions)
# - training_curves.png (with best epoch marker)
# - train_test_comparison.png (performance comparison)
# - Individual metric plots for both training and test sets

# 2. Graph + clinical features - BEST MODEL
python GNNClassificationBuilder.py \
    --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
    --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
    --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_graph_clinical \
    --gpu 0 \
    --gnn_type GAT \
    --hidden_dim 64 \
    --num_layers 3 \
    --num_heads 4 \
    --pooling mean \
    --batch_size 8 \
    --epochs 200 \
    --use_clinical

# 3. With GCN (memory efficient)
python GNNClassificationBuilder.py \
    --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
    --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
    --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_gcn \
    --gpu 0 \
    --gnn_type GCN \
    --hidden_dim 128 \
    --num_layers 3 \
    --pooling mean \
    --batch_size 8 \
    --epochs 100 \
    --use_clinical

# 4. Limited nodes per graph
python GNNClassificationBuilder.py \
    --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
    --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
    --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_limited_nodes \
    --gpu 0 \
    --gnn_type GAT \
    --hidden_dim 128 \
    --num_layers 3 \
    --num_heads 4 \
    --pooling mean \
    --batch_size 8 \
    --max_nodes 2000 \
    --epochs 100 \
    --use_clinical

# 5. Using 'both' pooling (mean + max)
python GNNClassificationBuilder.py \
    --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
    --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
    --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_both_pooling \
    --gpu 0 \
    --gnn_type GAT \
    --hidden_dim 128 \
    --num_layers 3 \
    --num_heads 4 \
    --pooling both \
    --batch_size 8 \
    --epochs 100 \
    --use_clinical
"""
        
# # 1. Graph features only (no clinical features)
# CUDA_VISIBLE_DEVICES=2 python GNNClassificationBuilder.py \
#     --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
#     --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
#     --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_graph_only \
#     --gpu 2 \
#     --gnn_type GCN \
#     --hidden_dim 128 \
#     --num_layers 3 \
#     --num_heads 4 \
#     --pooling mean \
#     --batch_size 8 \
#     --epochs 100

# # 2. Graph + clinical features
# python GNNClassificationBuilder.py \
#     --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
#     --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
#     --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_graph_clinical \
#     --gpu 0 \
#     --gnn_type GAT \
#     --hidden_dim 128 \
#     --num_layers 3 \
#     --num_heads 4 \
#     --pooling mean \
#     --batch_size 8 \
#     --epochs 100 \
#     --use_clinical

# # 3. With GCN instead of GAT (uses less memory)
# python GNNClassificationBuilder.py \
#     --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
#     --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
#     --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_gcn \
#     --gpu 0 \
#     --gnn_type GCN \
#     --hidden_dim 128 \
#     --num_layers 3 \
#     --pooling mean \
#     --batch_size 8 \
#     --epochs 100 \
#     --use_clinical

# # 4. With limited nodes per graph (for memory efficiency)
# python GNNClassificationBuilder.py \
#     --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
#     --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
#     --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_limited_nodes \
#     --gpu 0 \
#     --gnn_type GAT \
#     --hidden_dim 128 \
#     --num_layers 3 \
#     --num_heads 4 \
#     --pooling mean \
#     --batch_size 8 \
#     --max_nodes 2000 \
#     --epochs 100 \
#     --use_clinical

# # 5. Using 'both' pooling (mean + max)
# python GNNClassificationBuilder.py \
#     --graph_dir XXXyour path/Dataset/Dataset002_ECResponse/graphs/ \
#     --clinical_xlsx XXXyour path/Dataset/Dataset002_ECResponse/PatientLabels.xlsx \
#     --output_dir XXXyour path/Dataset/Dataset002_ECResponse/results_both_pooling \
#     --gpu 0 \
#     --gnn_type GAT \
#     --hidden_dim 128 \
#     --num_layers 3 \
#     --num_heads 4 \
#     --pooling both \
#     --batch_size 8 \
#     --epochs 100 \
#     --use_clinical