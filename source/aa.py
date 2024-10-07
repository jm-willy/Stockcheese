# Initialization Parameters for tf.keras.layers.MultiHeadAttention
mha = tf.keras.layers.MultiHeadAttention(
    num_heads=8,  # Number of attention heads
    key_dim=64,  # Dimension of key/query vectors (per head)
    value_dim=None,  # Dimension of value vectors (if None, uses key_dim)
    dropout=0.0,  # Dropout rate for attention weights
    use_bias=True,  # Whether to use bias in projections
    output_shape=None,  # Output shape after projection (if None, maintains last input dim)
    attention_axes=None,  # Axes over which attention is applied (None = all axes except batch & heads)
    kernel_initializer="glorot_uniform",  # Initializer for weight matrices
    bias_initializer="zeros",  # Initializer for bias vectors
    kernel_regularizer=None,  # Regularizer for weight matrices
    bias_regularizer=None,  # Regularizer for bias vectors
    activity_regularizer=None,  # Regularizer for layer output
    kernel_constraint=None,  # Constraint for weight matrices
    bias_constraint=None,  # Constraint for bias vectors
)

# Call Arguments
output = mha(
    query,  # Query tensor of shape (batch_size, query_seq_len, query_dim)
    value,  # Value tensor of shape (batch_size, value_seq_len, value_dim)
    key=None,  # Key tensor (if None, value is used as key)
    attention_mask=None,  # Boolean mask to prevent attention to certain positions
    return_attention_scores=False,  # Whether to return attention scores
    training=None,  # Whether in training mode (for dropout)
)

# Practical Example
import tensorflow as tf

# Create sample data
batch_size = 32
seq_length = 50
embed_dim = 256

# Create input tensors
query = tf.random.normal([batch_size, seq_length, embed_dim])
value = tf.random.normal([batch_size, seq_length, embed_dim])

# Create attention mask (optional)
# 1 means attend, 0 means mask
mask = tf.ones((batch_size, seq_length, seq_length))

# Initialize MultiHeadAttention layer
mha = tf.keras.layers.MultiHeadAttention(
    num_heads=8,
    key_dim=32,  # dim_per_head = embed_dim // num_heads
    dropout=0.1,
)

# Call the layer
output_tensor, attention_scores = mha(
    query=query, value=value, attention_mask=mask, return_attention_scores=True
)

print(f"Output shape: {output_tensor.shape}")  # (batch_size, seq_length, embed_dim)
print(
    f"Attention scores shape: {attention_scores.shape}"
)  # (batch_size, num_heads, seq_length, seq_length)
