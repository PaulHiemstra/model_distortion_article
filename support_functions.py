from IPython.core.display import HTML
from pedalboard.io import AudioFile
from pedalboard import Pedalboard
import pandas as pd
import numpy as np

def wav_player(filepath):
    """ Adapted from: https://nbviewer.org/gist/Carreau/5507501/the%20sound%20of%20hydrogen.ipynb
    
    will display html 5 player for compatible browser

    Parameters :
    ------------
    filepath : relative filepath with respect to the notebook directory ( where the .ipynb are not cwd)
               of the file to play

    The browser need to know how to play wav through html5.

    there is no autoplay to prevent file playing when the browser opens
    """
    
    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>
    
    <body>
    <audio controls="controls" style="width:600px" >
      <source src="%s" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """%(filepath)
    display(HTML(src))

def read_proces_dump(input_wav, output_wav, effects):
    '''
    Reads a wav file, applies the specified effects and dumps the wav file again. 
    '''
    with AudioFile(input_wav) as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate
    effected = Pedalboard(effects)(audio, samplerate)
    with AudioFile(output_wav, 'w', samplerate, effected.shape[0]) as f:
        f.write(effected)

    assert audio.shape[0] == 1, 'The following code assumes mono sound, found stereo'
    return pd.DataFrame({'original': audio.flatten(), 
                'effected': effected.flatten(),
                'diff': (audio-effected).flatten()})

def dump_numpy_audio(output_data, reference_wav, output_wav):
    '''
    Dumps the numpy array in `ouput_data` as a wav file. The samplerate used is taken
    from the reference wav file, if the sound is stereo or mono from the shape of the array.
    Shape of the array should be `(1, size)` for mono sound and `(2, size)` for stereo sound. 
    If you pass a 1d array, we assume you want mono sound.  
    '''
    with AudioFile(reference_wav) as f:
        samplerate = f.samplerate

    with AudioFile(output_wav, 'w', samplerate, output_data.shape[0]) as f:
        f.write(output_data)
    
    return None

def resize_to_mono(nparray):
    '''
    Helper function that resizes the input `nparray` to a `(1,size)` mono array to be
    dumped as a wav file. 
    '''
    return np.resize(nparray, (1, nparray.size))

def plot_accuracy(history):
    return pd.DataFrame({"Accuracy":history.history['accuracy'], 
                         "Validation accuracy":history.history['val_accuracy']}).plot(xlabel = 'Epoch')

def plot_loss(history):
    return pd.DataFrame({"loss":history.history['loss'], 
                         "validation loss":history.history['val_loss']}).plot(xlabel = 'Epoch')