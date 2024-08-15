#!/bin/sh
##
# Collection of conventient functions to help review training data
#
# Load from a shell with:
#   . helpers.sh
#   OR
#   source helpers.sh
##

apr_help() {
	cat <<-EOF
	APR Commands:

	   apr_help	Display this help text

	   skim_wavs	Play all .wav files in directory, waiting for input between each
	  		Tip: Press-and-Hold <enter> for a few seconds to queue batches

	EOF
}

skim_wavs() {
	for audio_file in *.wav; do
		while :; do
			repeat='false'

			# Play audio file
			aplay "$audio_file"

			# Wait for command
			read -p '[n]ext [r]epeat, [d]elete, [s]top (n): ' cmd
			case "$cmd" in
				(n|N) : ;;
				(r|R) repeat='true';;
				(d|D) rm "$audio_file";;
				(s|S) return;;
			esac
			[ "$repeat" = 'false' ] && break
		done
	done
	echo 'Finished playing all .wav files in this directory!'
}


##
# Finished Loading
##

echo 'APR Helpers Loaded! Use the command "apr_help" to learn more.'
