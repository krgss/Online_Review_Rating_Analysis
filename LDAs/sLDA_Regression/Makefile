CC = g++



LSOURCE = ap.cpp cholesky.cpp cokus.cpp slda-alpha.cpp slda-data.cpp slda-estimate.cpp slda-inference.cpp slda-model.cpp spdinverse.cpp utils.cpp
LHEADER = ap.h apvt.h cholesky.h cokus.h nmath.h slda.h slda-alpha.h slda-data.h slda-estimate.h slda-inference.h slda-model.h spdinverse.h utils.h

slda: $(LSOURCE) $(HEADER)
	  $(CC) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	-rm -f *.o slda
