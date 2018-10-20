for f in $1/*; do
    if [[ $f == *"a."* ]]
    then
        python2 ./text_feats_to_hdf5_replacezero.py -n 10 $f $f ana
    fi
    if [[ $f == *"pw."* ]]
    then
        python2 ./text_feats_to_hdf5_replacezero.py -n 10 $f $f pw
    fi
done
