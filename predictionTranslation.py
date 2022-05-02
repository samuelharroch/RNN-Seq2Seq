from util import *

filename = "char2encoding.pkl"
sentence = "I slept very well."

#saveChar2encoding("char2encoding.pkl",input_token_index,16,71,reverse_target_char_index,num_decoder_tokens,target_token_index)

                    # 256                   89                                                  116
input_token_index, max_encoder_seq_length, max_encoder_seq_length, reverse_target_char_index, num_decoder_tokens, target_token_index = getChar2encoding(filename)

encoder_input_data = encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,max_encoder_seq_length)
# from the training part
encoder_model = load_model('encoder_modelPredTranslation.h5')
decoder_model = load_model('decoder_modelPredTranslation.h5')

# Turn our data into a binary vector.
input_seq = encoder_input_data

decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens, target_token_index, reverse_target_char_index)
print('-')
print('Input sentence:', sentence)
print('Decoded sentence:', decoded_sentence)
