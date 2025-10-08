# Freeze Training Examples

This directory contains examples for freeze (partial-parameter) fine-tuning.

## Qwen2.5-Coder Freeze Training

### Configuration File

`qwen2_5_coder_freeze_sft.yaml` - Freeze training configuration for Qwen2.5-Coder-1.5B-Instruct

### Key Parameters

- **Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Method**: Freeze training (only train the last 6 layers)
- **Extra Trainable Modules**: `embed_tokens`, `norm`
- **Dataset**: sentiment_clean
- **Template**: qwen
- **Precision**: bf16

### Freeze Training Strategy Explained

#### Current Configuration
```yaml
freeze_trainable_layers: 6
freeze_trainable_modules: all
freeze_extra_modules: embed_tokens,norm
```

**What this means:**
- ✅ **Last 6 layers trainable**: Only the final 6 transformer layers will be updated
- ✅ **All modules in those layers**: All components (attention, MLP, layer norms) in those 6 layers
- ✅ **Extra modules**: Additionally train `embed_tokens` (input embeddings) and `norm` (final layer norm)

#### Why This Configuration?

**Pros:**
- ✅ **Efficient**: Significantly reduces trainable parameters (~20-30% of full model)
- ✅ **Fast**: Faster training and lower memory usage
- ✅ **Effective**: Last layers capture task-specific patterns
- ✅ **Stable**: Preserves general knowledge in early layers

**When to use:**
- Domain adaptation (e.g., code → medical, general → finance)
- Task-specific fine-tuning with limited data
- Quick experimentation
- Resource-constrained environments

#### Alternative Configurations

**1. More Conservative (Fewer Trainable Layers)**
```yaml
freeze_trainable_layers: 3
freeze_trainable_modules: all
freeze_extra_modules: norm  # Only train final norm, not embeddings
```
- Use when: Very limited data, want to preserve more of base model
- Trainable params: ~10-15%

**2. More Aggressive (More Trainable Layers)**
```yaml
freeze_trainable_layers: 12
freeze_trainable_modules: all
freeze_extra_modules: embed_tokens,norm
```
- Use when: More data available, need more adaptation
- Trainable params: ~40-50%

**3. Selective Module Training**
```yaml
freeze_trainable_layers: 6
freeze_trainable_modules: mlp  # Only train MLP, freeze attention
freeze_extra_modules: norm
```
- Use when: Want to preserve attention patterns, only adapt feed-forward

**4. Minimal Training (Adapter-like)**
```yaml
freeze_trainable_layers: 0
freeze_trainable_modules: all
freeze_extra_modules: norm  # Only train final layer norm
```
- Use when: Extremely limited resources or data
- Trainable params: <1%

#### Recommended Settings by Model Size

**For Qwen2.5-Coder-1.5B (28 layers total):**
- **Light adaptation**: 3-4 layers (~10-15% params)
- **Standard adaptation**: 6-8 layers (~20-30% params) ← **Current config**
- **Heavy adaptation**: 12-16 layers (~40-60% params)

**General Rule of Thumb:**
- Small models (<3B): Train 20-40% of layers
- Medium models (3-10B): Train 15-30% of layers
- Large models (>10B): Train 10-20% of layers

#### Should You Include embed_tokens and norm?

**embed_tokens (Input Embeddings):**
- ✅ **Include if**: New domain vocabulary, different text style
- ❌ **Exclude if**: Same domain, want maximum stability
- **Current config**: ✅ Included (good for sentiment analysis)

**norm (Final Layer Normalization):**
- ✅ **Include**: Almost always recommended
- **Why**: Helps adapt output distribution to new task
- **Cost**: Minimal parameters, high impact
- **Current config**: ✅ Included (recommended)

#### Performance Expectations

**Current Configuration (6 layers + embed_tokens + norm):**
- Trainable parameters: ~25-30% of total
- Training speed: 2-3x faster than full fine-tuning
- Memory usage: ~40-50% of full fine-tuning
- Performance: 85-95% of full fine-tuning quality (task-dependent)

#### Validation Strategy

Monitor these metrics to adjust:
1. **Validation loss plateaus early** → Increase trainable layers
2. **Overfitting quickly** → Decrease trainable layers
3. **Poor task performance** → Add more layers or include embed_tokens
4. **Good performance** → Try reducing layers to save resources

### Usage

```bash
llamafactory-cli train examples/train_freeze/qwen2_5_coder_freeze_sft.yaml
```

### Notes

1. **freeze_trainable_layers: 6** - Only the last 6 layers will be trainable
2. **freeze_extra_modules** - Additionally trains embedding and normalization layers
3. **Dataset** - Make sure `sentiment_clean` is defined in `data/dataset_info.json`
4. **Evaluation** - Configured with 20% validation split and accuracy metric

### Expected Behavior

- Training will freeze most of the model parameters
- Only the last 6 transformer layers + embed_tokens + norm will be updated
- This significantly reduces memory usage and training time
- Suitable for domain adaptation or task-specific fine-tuning

### Troubleshooting

If you encounter module name errors, you can check the actual module names by:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True)
for name, _ in model.named_modules():
    print(name)
```

Common module names for Qwen models:
- `model.embed_tokens` - Token embeddings
- `model.norm` - Final layer normalization
- `model.layers.X` - Transformer layers (X = layer number)
