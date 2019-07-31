# voice-corpus-tool
A tool for creation, manipulation and maintenance of voice corpora

## Installation
The tool requires Python packages `pydub` and `intervaltree`.
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

The first command __add__ requires one parameter. In our case we pass `'/data/sample-00300*.mp3'` in apostrophes to ensure the shell is not resolving the asterisk, but just forwards it to the tool which will do the wildcard processing instead.
This operation adds all wildcard-matching samples to the virtual buffer. To document this fact, it prints "Added 10 samples to buffer.".

Now the second and third commands (__skip__ and __take__) and their respective output should explain themselves.

Finally the command __play__ results in playing all remaining samples of the buffer. As they were directly added as files, there is no transcript associated with them. If samples were loaded from a voice corpus CSV file (like provided by the Common Voice project), each (voice) sample would feature its transcript. This transcript will then be kept associated to its sample throughout all further 1-to-1 processing of this sample.

Be aware that the "buffer" is virtual in the sense of not loading any audio data into memory. Its purpose is just to assign operations to sequences of samples. Only final output commands like __write__ or __play__ and the command __augment__ result in actual sample processing (on a file by file basis).

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
	Drop all samples, whose transcription does not contain a keyword
	Arguments:
		keyword: string - Keyword to look for in transcriptions

  tagged <tag> 
	Keep only samples with a specific tag
	Arguments:
		tag: string - Tag to look for

  settag <tag> 
	Sets a tag on all samples of current buffer
	Arguments:
		tag: string - Tag to set

  clear  
	Clears sample buffer

Named buffers:

  set <name> [-percent <percent>]
	Replaces named buffer with portion of buffer
	Arguments:
		name: string - Name of the named buffer
	Options:
		-percent: int - Percentage of samples from the beginning of buffer. If omitted, complete buffer.

  stash <name> [-percent <percent>]
	Moves buffer portion to named buffer. Moved samples will not remain in main buffer.
	Arguments:
		name: string - Name of the named buffer
	Options:
		-percent: int - Percentage of samples from the beginning of buffer. If omitted, complete buffer.

  push <name> [-percent <percent>]
	Appends portion of buffer samples to named buffer
	Arguments:
		name: string - Name of the named buffer
	Options:
		-percent: int - Percentage of samples from the beginning of buffer. If omitted, complete buffer.

  slice <name> <percent> 
	Moves portion of named buffer to current buffer
	Arguments:
		name: string - Name of the named buffer
		percent: int - Percentage of samples from the beginning of named buffer

  drop <name> 
	Drops named buffer
	Arguments:
		name: string - Name of the named buffer

Output:

  print  
	Prints list of samples in current buffer

  play  
	Play samples of current buffer

  pipe  
	Pipe raw sample data of current buffer to stdout. Could be piped to "aplay -r 44100 -c 2 -t raw -f s16".

  write <dir_name> [-just_csv]
	Write samples of current buffer to disk
	Arguments:
		dir_name: string - Path to the new sample directory. The directory and a file with the same name plus extension ".csv" should not exist.
	Options:
		-just_csv: bool - Prevents writing samples

  hdf5 <alphabet_path> <hdf5_path> [-ninput <ninput>] [-ncontext <ncontext>]
	Write samples to hdf5 MFCC feature DB that can be used by DeepSpeech
	Arguments:
		alphabet_path: string - Path to DeepSpeech alphabet file to use for transcript mapping
		hdf5_path: string - Target path of hdf5 feature DB
	Options:
		-ninput: int - Number of MFCC features (defaults to 26)
		-ncontext: int - Number of frames in context window (defaults to 9)

Effects:

  compr <kbit> 
	Distortion by mp3 compression
	Arguments:
		kbit: int - Virtual bandwidth in kBit/s

  rate <rate> 
	Resampling to different sample rate
	Arguments:
		rate: int - Sample rate to apply

  augment <source> [-times <times>] [-gain <gain>]
	Augment samples of current buffer with noise
	Arguments:
		source: string - CSV file with samples to augment onto current sample buffer
	Options:
		-times: int - How often to apply the augmentation source to the sample buffer
		-gain: float - How much gain (in dB) to apply to augmentation audio before overlaying onto buffer samples
```


