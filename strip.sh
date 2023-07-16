#!/bin/bash

echo $1 > password.txt
pdftk exercise-encrypted.pdf input_pw $1 output exercise.pdf
