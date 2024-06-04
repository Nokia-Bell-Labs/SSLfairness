# MIMIC Selected Models
* SimCLR with finetuned dense layer (1): 'code/baselines/SimCLR/MIMIC/20230302-162816_l1_hs128_e50_esFalse_bs16/simclr.finetuned.0.90.hdf5' (trainable parameters: 194)
* SimCLR with finetuned dense layers with hidden size 128 (2): 'code/baselines/SimCLR/MIMIC/20230302-163532_l2_hs128_e50_esFalse_bs16/simclr.finetuned.0.90.hdf5' (trainable parameters: 12,674)
* SimCLR with finetuned dense layers with hidden size 128 (3): 'code/baselines/SimCLR/MIMIC/20230302-163814_l3_hs128_e50_esFalse_bs16/simclr.finetuned.0.90.hdf5' (trainable parameters: 29,186) &rarr; Assigns everything to majority class
* SimCLR with fintuned dense layer(s) + 1/3 Conv. layer: '' (trainable parameters:)
* SimCLR with fintuned dense layer(s) + 2/3 Conv. layers: '' (trainable parameters:)
* SimCLR with fintuned dense layer(s) + 3/3 Conv. layers (technically supervised): 'code/baselines/SimCLR/MIMIC/20230302-164518_fr0/simclr.0.88.none_frozen.hdf5' (trainable parameters: 153,154)
