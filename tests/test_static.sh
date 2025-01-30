test_output_file=test_output.txt

python simulation.py --tlc_M 31 --tlc_K 2 --output_file $test_output_file
python simulation.py --tlc_M 30 --tlc_K 2 --output_file $test_output_file
python simulation.py --tlc_M 29 --tlc_K 2 --output_file $test_output_file
python simulation.py --tlc_M 28 --tlc_K 2 --output_file $test_output_file
python simulation.py --tlc_M 27 --tlc_K 2 --output_file $test_output_file
python simulation.py --num_tlc_ssds 0 --qlc_M 29 --qlc_K 1 --output_file $test_output_file
python simulation.py --num_tlc_ssds 0 --qlc_M 14 --qlc_K 2 --output_file $test_output_file
python simulation.py --num_tlc_ssds 16 --tlc_M 15 --tlc_K 1 --qlc_M 15 --qlc_K 1 --output_file $test_output_file
python simulation.py --num_tlc_ssds 16 --tlc_M 14 --tlc_K 2 --qlc_M 14 --qlc_K 2 --output_file $test_output_file
python simulation.py --num_tlc_ssds 4 --tlc_M 3 --tlc_K 1 --qlc_M 26 --qlc_K 2 --output_file $test_output_file
python simulation.py --num_tlc_ssds 4 --tlc_M 3 --tlc_K 1 --qlc_M 26 --qlc_K 2 --output_file $test_output_file
python simulation.py --num_tlc_ssds 4 --tlc_M 15 --tlc_K 1 --qlc_M 26 --qlc_K 2 --mixed_redundancy --output_file $test_output_file
python simulation.py --num_tlc_ssds 4 --tlc_M 14 --tlc_K 2 --qlc_M 26 --qlc_K 2 --mixed_redundancy --output_file $test_output_file
python simulation.py --num_tlc_ssds 16 --tlc_M 15 --tlc_K 1 --qlc_M 15 --qlc_K 1 --mixed_redundancy --output_file $test_output_file
python simulation.py --num_tlc_ssds 16 --tlc_M 14 --tlc_K 2 --qlc_M 14 --qlc_K 2 --mixed_redundancy --output_file $test_output_file


