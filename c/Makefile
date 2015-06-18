all: bin_pixel_pairs.c
	python setup.py build_ext --inplace
debug: bin_pixel_pairs.c
	DEBUG=1 python setup.py build_ext --inplace

clean:
	rm bin_pixel_pairs.so
