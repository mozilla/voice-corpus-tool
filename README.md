# voice-corpus-tool
A tool for creation, manipulation and maintenance of voice corpora

## Installation
The tool requires Python packges `pydub` and `intervaltree`.
You can install them from the project root using the following command:
```
$ pip install -r requirements.txt
```

For processing samples also the `sox` command line tool is required. You can install it using your normal package manager or retrieve it from [here](http://sox.sourceforge.net/).

## Usage
Basic principle of the voice corpus tool is to apply a series of "commands" to a virtual buffer of samples. 

### Illustrating example
Imagine you have a folder full of audio samples. The following example shows how to play a bunch of them.

```
$ ./voice.py add '/data/sample-00300*.mp3' skip 2 take 3 play
Added 10 samples to buffer.
Removed first 2 samples from buffer.
Took 3 samples as new buffer.
Playing:
Filename: "/data/sample-003002.mp3"
Transcript: ""
Filename: "/data/sample-003003.mp3"
Transcript: ""
Filename: "/data/sample-003004.mp3"
Transcript: ""
Played 3 samples.
```

The first command __add__ requires one parameter. In our case we pass `'/data/sample-00300*.mp3'` in apostrophs to ensure the shell is not resolving the asterisk, but just forwards it to the tool which will do the wildcard processing instead.
This operation adds all wildcard-matching samples to the virtual buffer. To document this fact, it prints "Added 10 samples to buffer.".

Now the second and third commands (__skip__ and __take__) and their respective output should explain themselves.

Finally the command __play__ results in playing all remaining samples of the buffer. As they were directly added as files, there is no transcript associated with them. If samples were loaded from a voice corpus CSV file (like provided by the Common Voice project), each (voice) sample would feature its transcript. This transcript will then be kept associated to its sample throughout all further 1-to-1 processing of this sample.

Be aware that the "buffer" is virtual in the sense of not loading any audio data into memory. It's purpose is just to assign operations to sequences of samples. Only final output commands like __write__ or __play__ and the command __augment__ result in actual sample processing (on a file by file basis).

For getting a complete list of supported commands just use the help command like this:

```
$ ./voice.py help
A tool to apply a series of commands to a collection of samples.
Usage: voice.py (command <arg1> <arg2> ... [-opt1 [<value>]] [-opt2 [<value>]] ...)*

Commands:

  help  
	Display help message

  add <source> 
	Adds samples to current buffer
	Arguments:
		source: string - Name of a named buffer or filename of a CSV file or WAV file (wildcards supported)

Buffer operations:

  shuffle  
	Randoimize order of the sample buffer

  order  
	Order samples in buffer by length

  reverse  
	Reverse order of samples in buffer

  take <number> 
	Take given number of samples from the beginning of the buffer as new buffer
	Arguments:
		number: int - Number of samples

  repeat <number> 
	Repeat samples of current buffer <number> times as new buffer
	Arguments:
		number: int - How often samples of the buffer should get repeated

  skip <number> 
	Skip given number of samples from the beginning of current buffer
	Arguments:
		number: int - Number of samples

  find <keyword> 
	Drop all samples, who's transcription does not contain a keyword
	Arguments:
		keyword: string - Keyword to look for in transcriptions

  clear  
	Clears sample buffer

Named buffers:

  set <name> 
	Replaces named buffer with contents of buffer
	Arguments:
		name: string - Name of the named buffer

  stash <name> 
	Moves buffer to named buffer (buffer will be empty afterwards)
	Arguments:
		name: string - Name of the named buffer

  push <name> 
	Appends buffer to named buffer
	Arguments:
		name: string - Name of the named buffer

  drop <name> 
	Drops named buffer
	Arguments:
		name: string - Name of the named buffer

Output:

  print  
	Prints list of samples in current buffer

  play  
	Play samples of current buffer

  write <dir_name> 
	Write samples of current buffer to disk
	Arguments:
		dir_name: string - Path to the new sample directory. The directory and a file with the same name plus extension ".csv" should not exist.

Effects:

  reverb  [-room_scale <room_scale>] [-hf_damping <hf_damping>] [-wet_gain <wet_gain>] [-stereo_depth <stereo_depth>] [-reverberance <reverberance>] [-wet_only] [-pre_delay <pre_delay>]
	Adds reverberation to buffer samples
	Options:
		-room_scale: float - Room scale factor (between 0.0 to 1.0)
		-hf_damping: float - HF damping factor (between 0.0 to 1.0)
		-wet_gain: float - Wet gain in dB
		-stereo_depth: float - Stereo depth factor (between 0.0 to 1.0)
		-reverberance: float - Reverberance factor (between 0.0 to 1.0)
		-wet_only: bool - If to strip source signal on output
		-pre_delay: int - Pre delay in ms

  echo <gain_in> <gain_out> <delay_decay> 
	Adds an echo effect to buffer samples
	Arguments:
		gain_in: float - Gain in
		gain_out: float - Gain out
		delay_decay: string - Comma separated delay decay pairs - at least one (e.g. 10,0.1,20,0.2)

  speed <factor> 
	Adds an speed effect to buffer samples
	Arguments:
		factor: float - Speed factor to apply

  pitch <cents> 
	Adds a pitch effect to buffer samples
	Arguments:
		cents: int - Cents (100th of a semi-tome) of shift to apply

  tempo <factor> 
	Adds a tempo effect to buffer samples
	Arguments:
		factor: float - Tempo factor to apply

  sox <effect> <args> 
	Adds a SoX effect to buffer samples
	Arguments:
		effect: string - SoX effect name
		args: string - Comma separated list of SoX effect parameters (no white space allowed)

  augment <source> [-gain <gain>] [-times <times>]
	Augment samples of current buffer with noise
	Arguments:
		source: string - CSV file with samples to augment onto current sample buffer
	Options:
		-gain: float - How much gain (in dB) to apply to augmentation audio before overlaying onto buffer samples
		-times: int - How often to apply the augmentation source to the sample buffer
```


