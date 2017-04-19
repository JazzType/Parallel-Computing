#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <immintrin.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#define FIRST_FILE  0
#define SECOND_FILE 1
#define BOTH_FILES  2

/* Prototypes */
int is_equal(void *, void *);
void print(void *);
void print_block(void *);
size_t total[] = {0, 0, 0};
struct mmapfile {
	int fd;
	off_t fsize;
	char *memmptr;
};

struct blkdata {
	uint64_t nblks;
	int bused;
};

struct fparams {
	__m256i *blocksfile1;
	__m256i *blocksfile2;	
};

void numblks_req(int fsize, struct blkdata *fblk) {	
	if (fsize % 32 == 0) {
		fblk->nblks = fsize / 32;
		fblk->bused = 0;
	}
	else { 
		fblk->nblks = (int)(fsize/32) + 1;
		fblk->bused = fsize % 32;
	}
}

void compare_files(struct fparams fparam, int nblks, int s_offset, int e_offset) {
	__m256i result;
	__m256i metaresult;
	__m256i mask;	
	int equality;
	mask = _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1);	
	for(int iter = s_offset; iter < e_offset; iter++) {
		result = _mm256_cmpeq_epi32(fparam.blocksfile1[iter], fparam.blocksfile2[iter]);
		metaresult = _mm256_cmpeq_epi32(result, mask);
		equality = is_equal(&result, &mask);
		if(equality == -1) {			
			/* Line seen in both files */
			total[BOTH_FILES]++;			
		}
		/**
		* Any other case is not handled because focus of optimization was on 
		* vector comparison using AVX2. This scenario was being utilized only 
		* in the case when the two blocks loaded were being compared as a 256b 
		* vector. In case they were not the same, then a different mechanism 
		* has to be utilized which could not have been SIMD-oriented, since
		* comparison ahs to be done on a finer level.
		* To be as fair as possible to the original GNU comm program, 
		* all output conditions have been closely replicated, i.e. comm(1) no 
		* longer outputs the common portion of the file, and hence neither does this.
		* To prevent any print-related bottlenecks, all print calls have been 
		* commented out wherever possible, both in comm(1) and this AVX2 implementation.
		**/
	}
}

int is_equal(void *value1, void *value2) {
	long *ar1 = (long*) value1;
	long *ar2 = (long*) value2;
	int maxiter = 4;
	for(int index = 0; index < maxiter; index++) {
		if(ar1[index] != ar2[index])
			return index;
	}
	return -1;
}

void print(void* val) {
	int *value = (int*) val;
	printf("%08x %08x %08x %08x %08x %08x %08x %08x\n", value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7]);
}

void print_block(void *block) {
	char *blk = (char*) block;
	for(int i = 0; i < 32; i++)
		printf("%c", blk[i]);
	printf("\n");
}

int main(int argc, char const *argv[]) {
	if(argc != 3) {
		fprintf(stderr, "Usage: comm <file1> <file2> [flags]\n");
		exit(1);
	}

	char *mmfile1 = NULL;
	char *mmfile2 = NULL;

	struct stat statbuf;
	struct mmapfile *filearray = (struct mmapfile*) malloc(sizeof(struct mmapfile) * 2);

	/* Two files specified, now open and get details */
	for(int fileid = 1; fileid < argc; fileid++) {
		filearray[fileid-1].fd = open(argv[fileid], O_RDONLY);
		if(stat(argv[fileid], &statbuf) == -1) {
			fprintf(stderr, "stat error for file %d\n", fileid);
			_exit(1);
		}		
		filearray[fileid-1].fsize = statbuf.st_size;		
	}
	
	/* Attempt to map both open files to memory */
 	filearray[0].memmptr = (char*) mmap(NULL, filearray[0].fsize, PROT_READ, MAP_SHARED, filearray[0].fd, 0);
	filearray[1].memmptr = (char*) mmap(NULL, filearray[1].fsize, PROT_READ, MAP_SHARED, filearray[1].fd, 0);
	
	/* One or more mapping failed, exit */
	if(filearray[0].memmptr == MAP_FAILED) {
		fprintf(stderr, "Failed to load file into memory.\n");
		_exit(2);
	}
	if(filearray[1].memmptr == MAP_FAILED) {		
		fprintf(stderr, "Failed to load file into memory.\n");
		_exit(2);
	}	

	/**
	* Todo: AVX-512 for string comparison 
  *		? Scrapped, AVX-512 supported only on Knight's Landing and Xeon Phi series CPUs
	*		=> Using AVX2 instead, 256b instead of 512b
	**/	
	
	/* Initial indices */
	int idxs[8] = {0, 1, 2, 3, 4, 5, 6, 7}; 
	__m256i mask;
	mask = _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1);	
	__m256i file1, file2;
	__m256i indices = _mm256_maskload_epi32(idxs, mask);
	__m256i zero = _mm256_setzero_si256();
	__m256i mask7 = _mm256_set_epi32(8, 8, 8, 8, 8, 8, 8, 8);		
	
	/* Structures for getting block breakdown information */
	struct blkdata file1blkdata, file2blkdata;

	/**
	* Todo: Break up mapped string into 256b blocks
	* and calculate the number of blocks required.
	*  ? Done
	* => 
	**/
	numblks_req(filearray[0].fsize, &file1blkdata);
	numblks_req(filearray[1].fsize, &file2blkdata);
	//printf("Number of blocks required: %d %d\n", file1blkdata.nblks, file2blkdata.nblks);
	
	__m256i *blocksfile1;
	__m256i *blocksfile2;
	if((posix_memalign((void**)&blocksfile1, 32, sizeof(__m256i) * file1blkdata.nblks)) == 0) {
		if((posix_memalign((void**)&blocksfile2, 32, sizeof(__m256i) * file2blkdata.nblks)) == 0) {
			
		}
	}
	else {
		fprintf(stderr, "Could not obtain aligned memory\n");
		exit(-1);
	}

	//printf("Allocated file block arrays\n");
	//print(&blocksfile1[1]);
	int niter = file1blkdata.nblks < file2blkdata.nblks ? file2blkdata.nblks : file1blkdata.nblks;
	int blockno;
	for(blockno = 0; blockno < niter; blockno++) {
		//print(&indices);
		blocksfile1[blockno] = _mm256_i32gather_epi32(filearray[0].memmptr, indices, 4);
		blocksfile2[blockno] = _mm256_i32gather_epi32(filearray[1].memmptr, indices, 4);
		//print(&blocksfile2);
		//blocksfile2[blockno] = _mm256_setzero_si256();
		//printf("iter: %d\n", blockno);
		indices = _mm256_add_epi32(indices, mask7);
	}
	//printf("Standard mapping complete\n");
	/* Handle the remainder of blocks */
	if(file1blkdata.nblks < niter) {
		blocksfile2[blockno] = _mm256_i32gather_epi32(filearray[1].memmptr, indices, 4);		
		//print_block(&blocksfile1[blockno]);
	}
	else if(file2blkdata.nblks < niter) {
		blocksfile1[blockno] = _mm256_i32gather_epi32(filearray[0].memmptr, indices, 4);
		//print_block(&blocksfile2[blockno]);
	}
	/**
	* Todo: Split block array across threads
	**/
	int nthreads = 2; /* Soft limit */
	int nblkpthread = niter / nthreads;
	/* Everything till now was successful, begin actual process */
	/**
	* Todo: pthread implementation for work division
	**/
	struct fparams fparam;
	fparam.blocksfile1 = blocksfile1;
	fparam.blocksfile2 = blocksfile2;
	//printf("Now printing what is about to be sent\n");
	//print_block(&(fparam.blocksfile1[0]));
	compare_files(fparam, nblkpthread, 0, niter);
	free(blocksfile1);
	free(blocksfile2);
	munmap(filearray[0].memmptr, filearray[0].fsize);
	munmap(filearray[1].memmptr, filearray[1].fsize);
	free(filearray);
	return 0;
}