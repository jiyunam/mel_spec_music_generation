# mel_spec_music_generation

###About the Project

Since the rapid development of automation, the task of trying to algorithmically compose music with little human intervention has been of great interest. This interest is further elevated in the field of small to mid-sized game or film production companies that find the financial cost involved in commissioning original, high-quality background music difficult to meet. Thus, a need for algorithmic composition arises.

There has been past implementation that aimed to address this need, such as works that use MIDI-file inputs to apply a [Hierarchical Markov Model](https://www.aaai.org/ocs/index.php/AIIDE/AIIDE15/paper/view/11539/11364) in the task of algorithmic composition. While such applications that use symbolic representation of music has the benefit of using data without certain "noise" that can complicate the process of training, these elements of "noise" can include important musical details such as timbre or tone when it comes to generating music. There also exists industry applications, such as [Google DeepMind's 2016 WaveNet architecture](https://arxiv.org/pdf/1609.03499.pdf), but tend to be difficult to reproduce the same quality of results given their complexity and large datasets required to generate results.

Consequently, this project serves as a proof of concept of a much simpler model capable of generating automatically generated pieces and trained off of raw audio input. 


### Requires

- librosa
- numpy
- ffmpeg (for mp3 support backend) -> conda install -c conda-forge ffmpeg librosa