# Tree2Seq

## Tree2Seq in pytorch 
In this repository we implemented Tree2Seq and several baseline in pytorch (Version 1.0). To make the code more reusable we diveded each model in a separated files (obivuosly there is a large code overlap). In the folder models you can find the following:
- ****Tree2Seq*** Our model
- ***Mem2Seq***: Memory to Sequence 
- ***Seq2Seq***: Vanilla seq2seq model with no attention (enc_vanilla)
- ***+Attn***: Luong attention attention model
- ***Ptr-Unk***: combination between Bahdanau attention and Pointer Networks ([Point to UNK words](http://www.aclweb.org/anthology/P16-1014)) 

the option you can choose are:
- `-t` this is task dependent. 1-6 for bAbI and nothing for In-Car
- `-ds` choose which dataset to use (babi and kvr)
- `-dec` to choose the model. The option are: Mem2Seq, VanillaSeqToSeq, LuongSeqToSeq, PTRUNK
- `-hdd` hidden state size of the two rnn
- `-bsz` batch size
- `-lr` learning rate
- `-dr` dropout rate
- `-layer` number of stacked rnn layers, or number of hops for Mem2Seq
- `-gpu_ranks` ids for gpu which can be used in the training procedure
- `-debug` option that switch the training procedure into debug mode (with smaller cost and just one process for the ease of debugging.)
- `-experiment` the name for the experiment, which will be used in saving the model file and the log file.

## Run the script
To start the model training procedure, you can use the following command. 
You can adjust some of the options in the script to make it suitable for your computer.

If you set the ***gpu_ranks***, the experiment will be running on distributed mode automatically.

> bash train_kvr_tree2seq.sh 



