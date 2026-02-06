import streamlit as st
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure TensorFlow to be more stable
tf.config.experimental.enable_op_determinism()

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with caching to avoid repeated loading"""
    try:
        # Clear session for clean state
        tf.keras.backend.clear_session()
        
        # Load model without compiling to avoid optimizer issues
        model = load_model("next_word_generator.keras", compile=False)
        
        # Load tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        
        return model, tokenizer, True, "Model loaded successfully!"
    
    except Exception as e:
        return None, None, False, f"Error loading model: {str(e)}"

def safe_tokenize(tokenizer, text):
    """Safely tokenize text with error handling"""
    try:
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = text.strip()
        
        # Convert to sequences
        sequences = tokenizer.texts_to_sequences([text])
        
        if not sequences or not sequences[0]:
            # If tokenization fails, try with individual words
            words = text.lower().split()
            sequence = []
            for word in words:
                if word in tokenizer.word_index:
                    sequence.append(tokenizer.word_index[word])
            return sequence if sequence else [1]  # Return [1] as fallback (usually <unk> token)
        
        return sequences[0]
    
    except Exception as e:
        st.warning(f"Tokenization issue: {str(e)}. Using fallback.")
        return [1]  # Fallback to unknown token

def safe_predict(model, sequence, max_len=20):
    """Safely make predictions with proper error handling"""
    try:
        if not sequence:
            sequence = [1]  # Fallback sequence
        
        # Ensure sequence is not too long
        if len(sequence) > max_len:
            sequence = sequence[-max_len:]
        
        # Pad sequence
        padded = pad_sequences([sequence], maxlen=max_len, padding='pre', truncating='pre')
        
        # Make prediction with proper session management
        with tf.keras.backend.get_session().as_default() if hasattr(tf.keras.backend, 'get_session') else tf.device('/CPU:0'):
            predictions = model.predict(padded, verbose=0, batch_size=1)
        
        return predictions[0] if predictions is not None and len(predictions) > 0 else None
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def find_word_from_index(tokenizer, index):
    """Find word from token index with fallback"""
    try:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                return word
        return None
    except:
        return None

def generate_next_word(model, tokenizer, text, temperature=1.0):
    """Generate next word with temperature control for variety"""
    try:
        # Tokenize input
        sequence = safe_tokenize(tokenizer, text)
        
        if not sequence:
            return None, "Could not tokenize input text"
        
        # Get predictions
        predictions = safe_predict(model, sequence)
        
        if predictions is None:
            return None, "Prediction failed"
        
        # Apply temperature for variety
        if temperature > 0:
            predictions = np.asarray(predictions).astype('float64')
            predictions = np.log(predictions + 1e-8) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)
        
        # Choose next word index
        if temperature == 0:
            # Greedy selection
            next_index = np.argmax(predictions)
        else:
            # Probabilistic selection
            next_index = np.random.choice(len(predictions), p=predictions)
        
        # Find corresponding word
        next_word = find_word_from_index(tokenizer, next_index)
        
        if next_word is None:
            # Fallback: try top-k sampling
            top_k = 10
            top_indices = np.argsort(predictions)[-top_k:]
            for idx in reversed(top_indices):
                word = find_word_from_index(tokenizer, idx)
                if word:
                    return word, "Generated successfully"
            
            return None, f"Could not find word for any of top indices"
        
        return next_word, "Generated successfully"
    
    except Exception as e:
        return None, f"Generation error: {str(e)}"

# Streamlit App
st.title("🤖 Advanced Text Generator")
st.markdown("*Robust next-word prediction with error handling*")

# Load model and tokenizer
model, tokenizer, model_loaded, load_message = load_model_and_tokenizer()

if model_loaded:
    st.success(load_message)
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # User input with validation
        text = st.text_input(
            "Enter starting text:", 
            value="General Task Flow",
            help="Enter any text to start generation from"
        )
        
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        clear_btn = st.button("Clear Text", help="Clear the input field")
        if clear_btn:
            st.rerun()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        num_words = st.slider(
            "Number of words to generate:",
            min_value=1,
            max_value=100,
            value=10,
            help="How many words to generate"
        )
        
        generation_speed = st.slider(
            "Generation speed (seconds):",
            min_value=0.0,
            max_value=2.0,
            value=0.3,
            step=0.1,
            help="Delay between each word generation"
        )
        
        temperature = st.slider(
            "Creativity (temperature):",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher values = more creative/random, Lower values = more predictable"
        )
        
        st.markdown("---")
        show_debug = st.checkbox("Show debug info", help="Display technical information during generation")
    
    # Generation button
    if st.button("🚀 Generate Text", type="primary", help="Start generating text"):
        if not text or not text.strip():
            st.warning("⚠️ Please enter some starting text.")
        else:
            # Initialize generation
            generated_text = text.strip()
            
            # Create output containers
            output_container = st.empty()
            progress_container = st.empty()
            debug_container = st.empty() if show_debug else None
            
            # Progress tracking
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
            
            success_count = 0
            
            # Generation loop
            for i in range(num_words):
                try:
                    status_text.text(f"Generating word {i+1} of {num_words}...")
                    
                    # Generate next word
                    next_word, status = generate_next_word(model, tokenizer, generated_text, temperature)
                    
                    if debug_container:
                        debug_info = f"Step {i+1}: {status}"
                        if next_word:
                            debug_info += f" -> '{next_word}'"
                        debug_container.text(debug_info)
                    
                    if next_word:
                        generated_text += " " + next_word
                        success_count += 1
                        
                        # Update display
                        output_container.text_area(
                            "Generated Text:", 
                            generated_text, 
                            height=150,
                            key=f"output_{i}"
                        )
                        
                        # Update progress
                        progress_bar.progress((i + 1) / num_words)
                        
                        # Wait before next generation
                        if generation_speed > 0:
                            time.sleep(generation_speed)
                    else:
                        st.warning(f"⚠️ Could not generate word {i+1}: {status}")
                        if show_debug:
                            st.write(f"Debug: Current text length: {len(generated_text.split())} words")
                        break
                        
                except Exception as e:
                    st.error(f"❌ Error at word {i+1}: {str(e)}")
                    break
            
            # Final status
            progress_bar.progress(1.0)
            status_text.text(f"✅ Generated {success_count} words successfully!")
            
            if success_count > 0:
                st.success("🎉 Generation complete!")
                
                # Final output with copy option
                st.markdown("### Final Result:")
                st.text_area(
                    "Copy your generated text from here:", 
                    generated_text, 
                    height=100,
                    key="final_output"
                )
                
                # Statistics
                with st.expander("📊 Generation Statistics"):
                    st.write(f"- **Words generated:** {success_count}/{num_words}")
                    st.write(f"- **Success rate:** {success_count/num_words*100:.1f}%")
                    st.write(f"- **Final text length:** {len(generated_text.split())} words")
                    st.write(f"- **Characters:** {len(generated_text)}")
            else:
                st.error("❌ No words were generated successfully.")

    # Help section
    with st.expander("ℹ️ How to Use & Troubleshooting"):
        st.markdown("""
        ### How to Use:
        1. **Enter starting text** - Any text you want to continue from
        2. **Adjust settings** in the sidebar:
           - **Number of words**: How many words to generate
           - **Speed**: Delay between words (0 = instant)
           - **Creativity**: Higher = more random, Lower = more predictable
        3. **Click Generate** to start the process
        
        ### Troubleshooting:
        - **Model errors**: Try shorter input text or fewer generation words
        - **Slow generation**: Reduce generation speed or number of words
        - **Repetitive output**: Increase creativity (temperature)
        - **Random output**: Decrease creativity (temperature)
        
        ### Tips:
        - Start with text similar to your training data for best results
        - Use debug mode to see what's happening during generation
        - Experiment with different creativity levels for varied outputs
        """)

else:
    st.error("❌ " + load_message)
    
    with st.expander("🔧 Troubleshooting Steps"):
        st.markdown("""
        **Check these items:**
        1. ✅ Files exist: `next_word_generator.keras` and `tokenizer.pkl`
        2. ✅ Files are in the same directory as this script
        3. ✅ TensorFlow/Keras versions are compatible
        4. ✅ Model was saved properly without optimizer state
        
        **Try these solutions:**
        1. Restart Streamlit application
        2. Clear browser cache and reload
        3. Re-save your model using: `model.save('model.keras', include_optimizer=False)`
        4. Check file permissions
        """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and TensorFlow/Keras*")