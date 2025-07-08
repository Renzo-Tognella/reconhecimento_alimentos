import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import random
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score,
    accuracy_score, cohen_kappa_score, matthews_corrcoef
)
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÕES AVANÇADAS PARA ALIMENTOS
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16  # Menor para treinar com mais filtros
EPOCHS = 40      # Mais épocas
LEARNING_RATE = 0.0003  # Learning rate menor

# Caminhos das pastas
DATA_DIR_ORIGINAL = "Imagens_um_Alimento"

# Configuração para salvar métricas
METRICS_DIR = "metricas_diagnostico"
os.makedirs(METRICS_DIR, exist_ok=True)

class MetricsAnalyzer:
    """Classe para análise completa de métricas da CNN"""
    
    def __init__(self, model, X_test, y_test, class_names):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.y_pred_proba = None
        self.y_pred = None
        self.metrics = {}
        
    def compute_predictions(self):
        """Computa predições"""
        print(" Computando predições...")
        self.y_pred_proba = self.model.predict(self.X_test)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        
    def compute_basic_metrics(self):
        """Métricas básicas"""
        print("Computando métricas básicas...")
        
        # Accuracy
        self.metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        
        # F1 Score (macro, micro, weighted)
        self.metrics['f1_macro'] = f1_score(self.y_test, self.y_pred, average='macro')
        self.metrics['f1_micro'] = f1_score(self.y_test, self.y_pred, average='micro')
        self.metrics['f1_weighted'] = f1_score(self.y_test, self.y_pred, average='weighted')
        
        # Precision
        self.metrics['precision_macro'] = precision_score(self.y_test, self.y_pred, average='macro')
        self.metrics['precision_micro'] = precision_score(self.y_test, self.y_pred, average='micro')
        self.metrics['precision_weighted'] = precision_score(self.y_test, self.y_pred, average='weighted')
        
        # Recall
        self.metrics['recall_macro'] = recall_score(self.y_test, self.y_pred, average='macro')
        self.metrics['recall_micro'] = recall_score(self.y_test, self.y_pred, average='micro')
        self.metrics['recall_weighted'] = recall_score(self.y_test, self.y_pred, average='weighted')
        
        # Cohen's Kappa
        self.metrics['cohen_kappa'] = cohen_kappa_score(self.y_test, self.y_pred)
        
        # Matthews Correlation Coefficient
        self.metrics['matthews_corrcoef'] = matthews_corrcoef(self.y_test, self.y_pred)
        
        # Balanced Accuracy
        from sklearn.metrics import balanced_accuracy_score
        self.metrics['balanced_accuracy'] = balanced_accuracy_score(self.y_test, self.y_pred)
        
        # Hamming Loss
        from sklearn.metrics import hamming_loss
        self.metrics['hamming_loss'] = hamming_loss(self.y_test, self.y_pred)
        
        # Jaccard Score
        from sklearn.metrics import jaccard_score
        self.metrics['jaccard_macro'] = jaccard_score(self.y_test, self.y_pred, average='macro')
        self.metrics['jaccard_micro'] = jaccard_score(self.y_test, self.y_pred, average='micro')
        self.metrics['jaccard_weighted'] = jaccard_score(self.y_test, self.y_pred, average='weighted')
        
        # Zero-One Loss
        from sklearn.metrics import zero_one_loss
        self.metrics['zero_one_loss'] = zero_one_loss(self.y_test, self.y_pred)
        
        # Top-K Accuracy (se aplicável)
        if len(np.unique(self.y_test)) > 2:  # Para classificação multiclasse
            from sklearn.metrics import top_k_accuracy_score
            try:
                self.metrics['top_2_accuracy'] = top_k_accuracy_score(self.y_test, self.y_pred_proba, k=2)
                self.metrics['top_3_accuracy'] = top_k_accuracy_score(self.y_test, self.y_pred_proba, k=3)
            except:
                pass
        
    def compute_per_class_metrics(self):
        """Métricas por classe"""
        print("Computando métricas por classe...")
        
        report = classification_report(self.y_test, self.y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        self.metrics['per_class'] = report
        
        # F1 por classe
        f1_per_class = f1_score(self.y_test, self.y_pred, average=None)
        self.metrics['f1_per_class'] = dict(zip(self.class_names, f1_per_class))
        
        # Precision por classe
        precision_per_class = precision_score(self.y_test, self.y_pred, average=None)
        self.metrics['precision_per_class'] = dict(zip(self.class_names, precision_per_class))
        
        # Recall por classe
        recall_per_class = recall_score(self.y_test, self.y_pred, average=None)
        self.metrics['recall_per_class'] = dict(zip(self.class_names, recall_per_class))
        
    def plot_confusion_matrix(self):
        """Plota matriz de confusão"""
        print("🎯 Gerando matriz de confusão...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
        plt.xlabel('Predição', fontsize=14)
        plt.ylabel('Real', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/matriz_confusao.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Matriz de confusão normalizada
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusão Normalizada', fontsize=16, fontweight='bold')
        plt.xlabel('Predição', fontsize=14)
        plt.ylabel('Real', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/matriz_confusao_normalizada.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_roc_curves(self):
        """Plota curvas ROC para cada classe"""
        print("Gerando curvas ROC...")
        
        n_classes = len(self.class_names)
        
        # Calcular ROC para cada classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((self.y_test == i).astype(int), self.y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot individual ROC curves
        plt.figure(figsize=(15, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos', fontsize=14)
        plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=14)
        plt.title('Curvas ROC por Classe', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/curvas_roc.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # AUC médio
        self.metrics['auc_mean'] = np.mean(list(roc_auc.values()))
        self.metrics['auc_per_class'] = roc_auc
        
    def plot_precision_recall_curves(self):
        """Plota curvas Precision-Recall"""
        print("📊 Gerando curvas Precision-Recall...")
        
        n_classes = len(self.class_names)
        
        plt.figure(figsize=(15, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(
                (self.y_test == i).astype(int), self.y_pred_proba[:, i]
            )
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Curvas Precision-Recall por Classe', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/curvas_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_per_class_metrics(self):
        """Plota métricas por classe"""
        print("📊 Gerando gráficos de métricas por classe...")
        
        # F1, Precision, Recall por classe
        metrics_df = pd.DataFrame({
            'F1-Score': list(self.metrics['f1_per_class'].values()),
            'Precision': list(self.metrics['precision_per_class'].values()),
            'Recall': list(self.metrics['recall_per_class'].values())
        }, index=self.class_names)
        
        # Gráfico de barras
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        metrics_df['F1-Score'].plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('F1-Score por Classe', fontweight='bold')
        axes[0].set_ylabel('F1-Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        metrics_df['Precision'].plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Precision por Classe', fontweight='bold')
        axes[1].set_ylabel('Precision')
        axes[1].tick_params(axis='x', rotation=45)
        
        metrics_df['Recall'].plot(kind='bar', ax=axes[2], color='salmon')
        axes[2].set_title('Recall por Classe', fontweight='bold')
        axes[2].set_ylabel('Recall')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/metricas_por_classe.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Heatmap das métricas
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Valor'})
        plt.title('Métricas por Classe - Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/heatmap_metricas.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_training_history(self, history):
        """Plota histórico de treinamento"""
        print("Gerando gráficos do histórico de treinamento...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Treino')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validação')
        axes[0, 0].set_title('Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Treino')
        axes[0, 1].plot(history.history['val_loss'], label='Validação')
        axes[0, 1].set_title('Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate (se disponível)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate', fontweight='bold')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Diferença entre treino e validação
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        acc_diff = [t - v for t, v in zip(train_acc, val_acc)]
        
        axes[1, 1].plot(acc_diff, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting (Treino - Validação)', fontweight='bold')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Diferença de Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/historico_treinamento.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_error_analysis(self):
        """Análise de erros"""
        print(" Analisando erros...")
        
        # Encontrar predições incorretas
        incorrect_mask = self.y_test != self.y_pred
        incorrect_indices = np.where(incorrect_mask)[0]
        
        if len(incorrect_indices) == 0:
            print("✅ Nenhum erro encontrado!")
            return
        
        # Análise de erros por classe
        error_analysis = {}
        for i in range(len(self.class_names)):
            class_errors = []
            for idx in incorrect_indices:
                if self.y_test[idx] == i:
                    predicted_class = self.y_pred[idx]
                    confidence = self.y_pred_proba[idx][predicted_class]
                    class_errors.append({
                        'predicted': predicted_class,
                        'confidence': confidence,
                        'true_class': i
                    })
            error_analysis[i] = class_errors
        
        # Gráfico de erros por classe
        error_counts = [len(error_analysis[i]) for i in range(len(self.class_names))]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(self.class_names)), error_counts, color='red', alpha=0.7)
        plt.title('Número de Erros por Classe', fontsize=16, fontweight='bold')
        plt.xlabel('Classe Real', fontsize=14)
        plt.ylabel('Número de Erros', fontsize=14)
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45, ha='right')
        
        # Adicionar valores nas barras
        for bar, count in zip(bars, error_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/erros_por_classe.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribuição de confiança para erros
        all_error_confidences = []
        for errors in error_analysis.values():
            for error in errors:
                all_error_confidences.append(error['confidence'])
        
        if all_error_confidences:
            plt.figure(figsize=(10, 6))
            plt.hist(all_error_confidences, bins=20, color='red', alpha=0.7, edgecolor='black')
            plt.title('Distribuição de Confiança para Predições Incorretas', fontsize=16, fontweight='bold')
            plt.xlabel('Confiança da Predição', fontsize=14)
            plt.ylabel('Frequência', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{METRICS_DIR}/distribuicao_confianca_erros.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Matriz de confusão de erros (apenas erros)
        error_confusion = np.zeros((len(self.class_names), len(self.class_names)))
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j:  # Apenas erros
                    mask = (self.y_test == i) & (self.y_pred == j)
                    error_confusion[i, j] = np.sum(mask)
        
        if np.sum(error_confusion) > 0:
            plt.figure(figsize=(12, 10))
            sns.heatmap(error_confusion, annot=True, fmt='.0f', cmap='Reds',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title('Matriz de Confusão - Apenas Erros', fontsize=16, fontweight='bold')
            plt.xlabel('Predição', fontsize=14)
            plt.ylabel('Real', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'{METRICS_DIR}/matriz_confusao_erros.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Análise de confiança geral
        all_confidences = np.max(self.y_pred_proba, axis=1)
        correct_confidences = all_confidences[self.y_test == self.y_pred]
        incorrect_confidences = all_confidences[self.y_test != self.y_pred]
        
        plt.figure(figsize=(12, 6))
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Corretas', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorretas', color='red')
        plt.title('Distribuição de Confiança: Corretas vs Incorretas', fontsize=16, fontweight='bold')
        plt.xlabel('Confiança da Predição', fontsize=14)
        plt.ylabel('Frequência', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/distribuicao_confianca_geral.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calcular métricas de confiança
        self.metrics['confidence_analysis'] = {
            'mean_confidence_correct': np.mean(correct_confidences),
            'mean_confidence_incorrect': np.mean(incorrect_confidences),
            'std_confidence_correct': np.std(correct_confidences),
            'std_confidence_incorrect': np.std(incorrect_confidences),
            'confidence_threshold_95': np.percentile(correct_confidences, 5),  # 95% das corretas acima deste valor
            'confidence_threshold_99': np.percentile(correct_confidences, 1),  # 99% das corretas acima deste valor
        }
        
    def save_metrics_report(self):
        """Salva relatório completo de métricas"""
        print("💾 Salvando relatório de métricas...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar métricas em JSON
        import json
        with open(f'{METRICS_DIR}/metricas_completas_{timestamp}.json', 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Salvar relatório em texto
        with open(f'{METRICS_DIR}/relatorio_metricas_{timestamp}.txt', 'w') as f:
            f.write("RELATÓRIO COMPLETO DE MÉTRICAS - CNN ALIMENTOS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MÉTRICAS GLOBAIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {self.metrics['accuracy']:.4f}\n")
            f.write(f"F1-Score (Macro): {self.metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Micro): {self.metrics['f1_micro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {self.metrics['f1_weighted']:.4f}\n")
            f.write(f"Precision (Macro): {self.metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {self.metrics['recall_macro']:.4f}\n")
            f.write(f"Cohen's Kappa: {self.metrics['cohen_kappa']:.4f}\n")
            f.write(f"Matthews Correlation: {self.metrics['matthews_corrcoef']:.4f}\n")
            f.write(f"AUC Médio: {self.metrics['auc_mean']:.4f}\n\n")
            
            f.write("MÉTRICAS POR CLASSE:\n")
            f.write("-" * 20 + "\n")
            for class_name in self.class_names:
                f.write(f"\n{class_name}:\n")
                f.write(f"  F1-Score: {self.metrics['f1_per_class'][class_name]:.4f}\n")
                f.write(f"  Precision: {self.metrics['precision_per_class'][class_name]:.4f}\n")
                f.write(f"  Recall: {self.metrics['recall_per_class'][class_name]:.4f}\n")
                f.write(f"  AUC: {self.metrics['auc_per_class'][self.class_names.index(class_name)]:.4f}\n")
            
            f.write(f"\n\nRelatório gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ Relatórios salvos em {METRICS_DIR}/")
        
    def generate_all_metrics(self, history):
        print("\n" + "="*60)
        print(" ANÁLISE COMPLETA DE MÉTRICAS")
        print("="*60)
        
        self.compute_predictions()
        self.compute_basic_metrics()
        self.compute_per_class_metrics()
        
        # Gerar gráficos
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_per_class_metrics()
        self.plot_training_history(history)
        self.plot_error_analysis()
        
        # Gerar dashboard completo
        self.generate_dashboard()
        
        # Salvar relatórios
        self.save_metrics_report()
        
        # Mostrar resumo
        print(f"\n📊 RESUMO DAS MÉTRICAS:")
        print(f"   Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"   F1-Score (Macro): {self.metrics['f1_macro']:.4f}")
        print(f"   Precision (Macro): {self.metrics['precision_macro']:.4f}")
        print(f"   Recall (Macro): {self.metrics['recall_macro']:.4f}")
        print(f"   Cohen's Kappa: {self.metrics['cohen_kappa']:.4f}")
        print(f"   AUC Médio: {self.metrics['auc_mean']:.4f}")
        
        return self.metrics
    
    def generate_dashboard(self):
        """Gera um dashboard completo com todas as métricas"""
        print("📊 Gerando dashboard completo...")
        
        # Criar figura com subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Métricas principais
        ax1 = plt.subplot(4, 3, 1)
        metrics_summary = [
            self.metrics['accuracy'],
            self.metrics['f1_macro'],
            self.metrics['precision_macro'],
            self.metrics['recall_macro']
        ]
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        bars = ax1.bar(metric_names, metrics_summary, color=colors, alpha=0.8)
        ax1.set_title('Métricas Principais', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Valor')
        ax1.set_ylim(0, 1)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metrics_summary):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Métricas avançadas
        ax2 = plt.subplot(4, 3, 2)
        advanced_metrics = [
            self.metrics['cohen_kappa'],
            self.metrics['matthews_corrcoef'],
            self.metrics['balanced_accuracy'],
            self.metrics.get('auc_mean', 0)
        ]
        advanced_names = ['Kappa', 'Matthews', 'Bal. Acc', 'AUC']
        
        bars = ax2.bar(advanced_names, advanced_metrics, color='#6A994E', alpha=0.8)
        ax2.set_title('Métricas Avançadas', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Valor')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars, advanced_metrics):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. F1-Score por classe
        ax3 = plt.subplot(4, 3, 3)
        f1_scores = list(self.metrics['f1_per_class'].values())
        bars = ax3.bar(range(len(self.class_names)), f1_scores, color='#BC4749', alpha=0.8)
        ax3.set_title('F1-Score por Classe', fontweight='bold', fontsize=14)
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(range(len(self.class_names)))
        ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # 4. Precision vs Recall por classe
        ax4 = plt.subplot(4, 3, 4)
        precision_scores = list(self.metrics['precision_per_class'].values())
        recall_scores = list(self.metrics['recall_per_class'].values())
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax4.bar(x + width/2, recall_scores, width, label='Recall', alpha=0.8)
        
        ax4.set_title('Precision vs Recall por Classe', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Valor')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax4.legend()
        
        # 5. AUC por classe
        ax5 = plt.subplot(4, 3, 5)
        auc_scores = [self.metrics['auc_per_class'][i] for i in range(len(self.class_names))]
        bars = ax5.bar(range(len(self.class_names)), auc_scores, color='#F4A261', alpha=0.8)
        ax5.set_title('AUC por Classe', fontweight='bold', fontsize=14)
        ax5.set_ylabel('AUC')
        ax5.set_xticks(range(len(self.class_names)))
        ax5.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # 6. Distribuição de confiança
        ax6 = plt.subplot(4, 3, 6)
        if 'confidence_analysis' in self.metrics:
            conf_analysis = self.metrics['confidence_analysis']
            conf_metrics = [
                conf_analysis['mean_confidence_correct'],
                conf_analysis['mean_confidence_incorrect'],
                conf_analysis['confidence_threshold_95'],
                conf_analysis['confidence_threshold_99']
            ]
            conf_names = ['Média Corretas', 'Média Incorretas', 'Threshold 95%', 'Threshold 99%']
            
            bars = ax6.bar(conf_names, conf_metrics, color='#E76F51', alpha=0.8)
            ax6.set_title('Análise de Confiança', fontweight='bold', fontsize=14)
            ax6.set_ylabel('Confiança')
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. Matriz de confusão (versão compacta)
        ax7 = plt.subplot(4, 3, 7)
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax7.imshow(cm_normalized, cmap='Blues', aspect='auto')
        ax7.set_title('Matriz de Confusão Normalizada', fontweight='bold', fontsize=14)
        ax7.set_xlabel('Predição')
        ax7.set_ylabel('Real')
        
        # Adicionar valores na matriz
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                text = ax7.text(j, i, f'{cm_normalized[i, j]:.2f}',
                               ha="center", va="center", color="black" if cm_normalized[i, j] < 0.5 else "white")
        
        # 8. Comparação de métricas (macro vs micro vs weighted)
        ax8 = plt.subplot(4, 3, 8)
        comparison_metrics = {
            'F1-Score': [self.metrics['f1_macro'], self.metrics['f1_micro'], self.metrics['f1_weighted']],
            'Precision': [self.metrics['precision_macro'], self.metrics['precision_micro'], self.metrics['precision_weighted']],
            'Recall': [self.metrics['recall_macro'], self.metrics['recall_micro'], self.metrics['recall_weighted']]
        }
        
        x = np.arange(3)
        width = 0.25
        
        for i, (metric, values) in enumerate(comparison_metrics.items()):
            ax8.bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        ax8.set_title('Comparação Macro/Micro/Weighted', fontweight='bold', fontsize=14)
        ax8.set_ylabel('Valor')
        ax8.set_xticks(x + width)
        ax8.set_xticklabels(['Macro', 'Micro', 'Weighted'])
        ax8.legend()
        
        # 9. Top-K Accuracy (se disponível)
        ax9 = plt.subplot(4, 3, 9)
        if 'top_2_accuracy' in self.metrics and 'top_3_accuracy' in self.metrics:
            top_k_metrics = [self.metrics['accuracy'], self.metrics['top_2_accuracy'], self.metrics['top_3_accuracy']]
            top_k_names = ['Top-1', 'Top-2', 'Top-3']
            
            bars = ax9.bar(top_k_names, top_k_metrics, color='#2A9D8F', alpha=0.8)
            ax9.set_title('Top-K Accuracy', fontweight='bold', fontsize=14)
            ax9.set_ylabel('Accuracy')
            ax9.set_ylim(0, 1)
            
            for bar, value in zip(bars, top_k_metrics):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 10. Resumo estatístico
        ax10 = plt.subplot(4, 3, 10)
        ax10.axis('off')
        
        summary_text = f"""
        RESUMO ESTATÍSTICO

        Métricas Principais:
        • Accuracy: {self.metrics['accuracy']:.4f}
        • F1-Score (Macro): {self.metrics['f1_macro']:.4f}
        • Precision (Macro): {self.metrics['precision_macro']:.4f}
        • Recall (Macro): {self.metrics['recall_macro']:.4f}

        Métricas Avançadas:
        • Cohen's Kappa: {self.metrics['cohen_kappa']:.4f}
        • Matthews Corr: {self.metrics['matthews_corrcoef']:.4f}
        • Balanced Acc: {self.metrics['balanced_accuracy']:.4f}
        • AUC Médio: {self.metrics.get('auc_mean', 0):.4f}

        Performance:
        • Melhor Classe: {max(self.metrics['f1_per_class'], key=self.metrics['f1_per_class'].get)} ({max(self.metrics['f1_per_class'].values()):.3f})
        • Pior Classe: {min(self.metrics['f1_per_class'], key=self.metrics['f1_per_class'].get)} ({min(self.metrics['f1_per_class'].values()):.3f})
        • Desvio Padrão F1: {np.std(list(self.metrics['f1_per_class'].values())):.3f}
        """
        
        ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 11. Distribuição de classes
        ax11 = plt.subplot(4, 3, 11)
        unique, counts = np.unique(self.y_test, return_counts=True)
        class_counts = [counts[list(unique).index(i)] for i in range(len(self.class_names))]
        
        bars = ax11.bar(range(len(self.class_names)), class_counts, color='#264653', alpha=0.8)
        ax11.set_title('Distribuição de Classes (Teste)', fontweight='bold', fontsize=14)
        ax11.set_ylabel('Número de Amostras')
        ax11.set_xticks(range(len(self.class_names)))
        ax11.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # 12. Performance vs Tamanho da Classe
        ax12 = plt.subplot(4, 3, 12)
        f1_vs_size = list(zip(class_counts, list(self.metrics['f1_per_class'].values())))
        f1_vs_size.sort(key=lambda x: x[0])  # Ordenar por tamanho
        
        sizes, f1s = zip(*f1_vs_size)
        ax12.scatter(sizes, f1s, s=100, alpha=0.7, c='#E63946')
        ax12.set_title('F1-Score vs Tamanho da Classe', fontweight='bold', fontsize=14)
        ax12.set_xlabel('Número de Amostras')
        ax12.set_ylabel('F1-Score')
        ax12.grid(True, alpha=0.3)
        
        # Adicionar linha de tendência
        if len(sizes) > 1:
            z = np.polyfit(sizes, f1s, 1)
            p = np.poly1d(z)
            ax12.plot(sizes, p(sizes), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{METRICS_DIR}/dashboard_completo.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Dashboard completo gerado!")

class FoodAugmentation:
    """Classe com filtros específicos para alimentos"""
    
    @staticmethod
    def simulate_different_lighting(image):
        """Simula diferentes condições de iluminação"""
        variations = []
        
        # 1. Iluminação mais quente (restaurante)
        warm = image.copy()
        warm[:,:,0] = np.clip(warm[:,:,0] * 1.1, 0, 1)  # Mais vermelho
        warm[:,:,1] = np.clip(warm[:,:,1] * 1.05, 0, 1) # Pouco mais verde
        variations.append(warm)
        
        # 2. Iluminação mais fria (luz fluorescente)
        cold = image.copy()
        cold[:,:,2] = np.clip(cold[:,:,2] * 1.1, 0, 1)  # Mais azul
        variations.append(cold)
        
        # 3. Iluminação dim (pouca luz)
        dim = image * 0.7
        dim = np.clip(dim, 0, 1)
        variations.append(dim)
        
        # 4. Iluminação bright (muita luz)
        bright = image * 1.3
        bright = np.clip(bright, 0, 1)
        variations.append(bright)
        
        return variations
    
    @staticmethod
    def simulate_camera_conditions(image):
        """Simula diferentes condições de câmera"""
        variations = []
        
        # 1. Simular blur (foto tremida)
        kernel_blur = np.ones((3,3), np.float32) / 9
        blurred = cv2.filter2D((image * 255).astype(np.uint8), -1, kernel_blur)
        variations.append(blurred.astype(np.float32) / 255.0)
        
        # 2. Simular noise (sensor ruim)
        noise = np.random.normal(0, 0.05, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        variations.append(noisy)
        
        # 3. Simular saturação alta (câmera saturada)
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 255)  # Mais saturação
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        variations.append(saturated.astype(np.float32) / 255.0)
        
        # 4. Simular baixo contraste
        low_contrast = 0.5 + (image - 0.5) * 0.6
        low_contrast = np.clip(low_contrast, 0, 1)
        variations.append(low_contrast)
        
        return variations
    
    @staticmethod
    def simulate_food_presentation(image):
        """Simula diferentes apresentações de comida"""
        variations = []
        
        # 1. Comida com vapor/umidade (blur sutil)
        steam_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16
        steamy = cv2.filter2D((image * 255).astype(np.uint8), -1, steam_kernel)
        variations.append(steamy.astype(np.float32) / 255.0)
        
        # 2. Comida ressecada (mais contraste, menos brilho)
        dry = np.clip((image - 0.5) * 1.2 + 0.4, 0, 1)
        variations.append(dry)
        
        # 3. Comida oleosa (mais brilho em pontos)
        oily = image.copy()
        # Adicionar pontos brilhantes aleatórios
        for _ in range(5):
            y, x = np.random.randint(20, IMG_HEIGHT-20), np.random.randint(20, IMG_WIDTH-20)
            oily[y-5:y+5, x-5:x+5] = np.clip(oily[y-5:y+5, x-5:x+5] * 1.5, 0, 1)
        variations.append(oily)
        
        return variations
    
    @staticmethod
    def apply_food_filters(image, num_variations=2):
        """Aplica filtros específicos para alimentos"""
        all_variations = [image]  # Incluir original
        
        # Aplicar diferentes tipos de filtros
        lighting_vars = FoodAugmentation.simulate_different_lighting(image)
        camera_vars = FoodAugmentation.simulate_camera_conditions(image)
        food_vars = FoodAugmentation.simulate_food_presentation(image)
        
        # Combinar todas as variações
        all_possible = lighting_vars + camera_vars + food_vars
        
        # Selecionar aleatoriamente algumas variações
        selected = random.sample(all_possible, min(num_variations, len(all_possible)))
        all_variations.extend(selected)
        
        return all_variations

def preprocess_food_image_advanced(image):
    """Pré-processamento específico para alimentos"""
    # 1. Converter para float32
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # 2. Redução de reflexos (comum em pratos)
    lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Reduzir highlights extremos
    l_clipped = np.clip(l, 0, 220)
    
    # 3. CLAHE para melhorar contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_enhanced = clahe.apply(l_clipped)
    
    # 4. Recompor
    enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    enhanced = enhanced.astype(np.float32) / 255.0
    
    # 5. Filtro bilateral para suavizar preservando bordas
    enhanced = cv2.bilateralFilter((enhanced * 255).astype(np.uint8), 5, 50, 50)
    enhanced = enhanced.astype(np.float32) / 255.0
    
    # 6. Sharpening muito sutil para texturas
    kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]]) * 0.05
    sharpened = cv2.filter2D((enhanced * 255).astype(np.uint8), -1, kernel_sharpen)
    result = cv2.addWeighted(
        (enhanced * 255).astype(np.uint8), 0.8,
        sharpened, 0.2, 0
    ).astype(np.float32) / 255.0
    
    return np.clip(result, 0, 1)

def load_data_with_advanced_filters():
    """Carrega dados com filtros para alimentos"""
    images = []
    labels = []
    class_names = []
    
    print(f" Carregando dados com FILTROS PARA ALIMENTOS...")
    
    # Obter classes
    data_dirs = [DATA_DIR_ORIGINAL]
    all_classes = set()
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for class_name in os.listdir(data_dir):
                class_path = os.path.join(data_dir, class_name)
                if os.path.isdir(class_path):
                    all_classes.add(class_name)
    
    class_names = sorted(list(all_classes))
    print(f"📊 Classes: {class_names}")
    
    total_original = 0
    total_augmented = 0
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
            
        dir_name = "Original" if "Imagens_um_Alimento" in data_dir else "Regiões"
        print(f"\n Processando {dir_name}...")
        
        for class_name in class_names:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_index = class_names.index(class_name)
            
            # Listar imagens
            class_images = []
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                class_images.extend([f for f in os.listdir(class_path) if f.endswith(ext)])
            
            if len(class_images) == 0:
                continue
            
            print(f"   {class_name}: {len(class_images)} imagens", end=" -> ")
            
            class_loaded = 0
            
            for img_name in class_images:
                img_path = os.path.join(class_path, img_name)
                
                try:
                    # Carregar imagem
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    
                    # Pré-processamento avançado
                    image = preprocess_food_image_advanced(image)
                    
                    # Adicionar original
                    images.append(image)
                    labels.append(class_index)
                    class_loaded += 1
                    total_original += 1
                    # APLICAR FILTROS AVANÇADOS
                    # Mais filtros para classes com poucas amostras
                    num_variations = 3 if len(class_images) < 50 else 2 if len(class_images) < 100 else 1
                    
                    if num_variations > 0:
                        variations = FoodAugmentation.apply_food_filters(image, num_variations)
                        
                        for var_img in variations[1:]:  # Pular original
                            if var_img.shape == (IMG_HEIGHT, IMG_WIDTH, 3):
                                images.append(var_img)
                                labels.append(class_index)
                                class_loaded += 1
                                total_augmented += 1
                
                except Exception as e:
                    print(f" Erro {img_name}: {e}")
                    continue
            
            print(f"{class_loaded} carregadas")
    
    print(f"\n RESUMO:")
    print(f"   Originais: {total_original}")
    print(f"   Augmentadas: {total_augmented}")
    print(f"   Total: {len(images)}")
    
    return np.array(images), np.array(labels), class_names

def create_robust_food_cnn(num_classes):
    """CNN robusta específica para alimentos"""
    
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Augmentation leve adicional (aplicada apenas no treino)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)  # Rotação menor
    x = layers.RandomZoom(0.05)(x)      # Zoom menor
    
    # Bloco 1: Detecção de bordas e texturas básicas
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Bloco 2: Texturas de alimentos
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Bloco 3: Padrões complexos
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Bloco 4: Features específicas de alimentos
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)
    
    # Bloco 5: Features muito específicas
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Camadas densas com regularização
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def main_advanced():
    print("TREINAMENTO CNN PARA ALIMENTOS")
    print("=" * 60)
    
    # Carregar dados com filtros avançados
    X, y, class_names = load_data_with_advanced_filters()
    
    if len(X) == 0:
        print(" Nenhuma imagem carregada!")
        return
    
    # Análise
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 DISTRIBUIÇÃO:")
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"  {class_names[class_idx]}: {count} imagens")
    
    # Dividir dados em treino/validação/teste (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 25% de 80% = 20% total
    )
    
    print(f"\n📊 DIVISÃO DOS DADOS:")
    print(f"   Treino: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validação: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Teste: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Criar modelo
    model = create_robust_food_cnn(len(class_names))
    
    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n Modelo criado:")
    model.summary()
    
    # Class weights
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Callbacks avançados
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'modelo_food_advanced.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Treinar
    print(f"\n Iniciando treinamento avançado ({EPOCHS} épocas)...")
    print("=" * 50)
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Avaliar no conjunto de validação
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n Resultado na validação:")
    print(f"    Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"    Loss: {val_loss:.4f}")
    
    # Avaliar no conjunto de teste
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n Resultado no teste:")
    print(f"    Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"    Loss: {test_loss:.4f}")
    
    # Salvar
    model.save('modelo_food_advanced_final.h5')
    
    with open('class_names_advanced.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"\n Modelos salvos!")
    print(f"   - modelo_food_advanced.h5 (melhor)")
    print(f"   - modelo_food_advanced_final.h5 (final)")
    print(" Treinamento avançado concluído!")

    # Gerar análise completa de métricas usando dados de teste
    print(f"\n" + "="*60)
    print(" INICIANDO ANÁLISE COMPLETA DE MÉTRICAS")
    print("="*60)
    
    metrics_analyzer = MetricsAnalyzer(model, X_test, y_test, class_names)
    metrics = metrics_analyzer.generate_all_metrics(history)
    
    # Salvar métricas finais
    final_metrics = {
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'detailed_metrics': metrics
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import json
    with open(f'{METRICS_DIR}/metricas_finais_{timestamp}.json', 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)
    
    print(f"\n🎉 ANÁLISE COMPLETA FINALIZADA!")
    print(f"📁 Todos os gráficos e relatórios salvos em: {METRICS_DIR}/")
    print(f"📊 Métricas finais salvas em: {METRICS_DIR}/metricas_finais_{timestamp}.json")

if __name__ == "__main__":
    main_advanced()