#!/bin/bash

dirs=('BPS_FH' 'POP909')

echo "Converting midi files to abc notation"

# Loop through every directory
for dir in ${dirs[@]}
do
    midi_dir="$dir"/midi
    output_dir="$dir"/abc/ 

    # Loop through every midi file in directory
    for midi in $midi_dir/*
    do
        echo $midi

        # Extract filename
        filename=$(echo $midi| cut -d'/' -f 3 | cut -d'.' -f 1)
        output_file="$output_dir$filename".abc
        echo $output_file

        midi2abc -f $midi > $output_file 
    done
done

echo "-------------------------------------------------------"
echo "Done!"
