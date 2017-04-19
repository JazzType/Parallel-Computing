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

size_t total[] = {0, 0, 0};
struct mmapfile {
	int fd;
	off_t fsize;
	char *memmptr;
};

struct blkdata {
	int nblks;
	int bused;
};
/* Prototypes */
void compare_files(struct mmapfile*);
void print(void *);
int is_equal(void*, void*);


int main(int argc, char const *argv[]) {
	if(argc != 3) {
		fprintf(stderr, "Usage: comm <file1> <file2> [flags]\n");
		exit(1);
	}


	char *mmfile1 = NULL;
	char *mmfile2 = NULL;

	struct stat statbuf;
	struct mmapfile *filearray = (struct mmapfile*) malloc(sizeof(struct mmapfile) * 2);

	//int *fda = malloc(sizeof(int) * (argc - 1));

	//off_t fsizes[] = {0, 0};

	/* Two files specified, now open and get details */
	for(int fileid = 1; fileid < argc; fileid++) {
		//fda[fileid] = open(argv[fileid], O_RDONLY);
		filearray[fileid-1].fd = open(argv[fileid], O_RDONLY);
		if(stat(argv[fileid], &statbuf) == -1) {
			fprintf(stderr, "stat error for file %d\n", fileid);
			_exit(1);
		}
		//fsizes[fileid-1] = statbuf.st_size;
		filearray[fileid-1].fsize = statbuf.st_size;
		//statbuf = NULL;
	}
	printf("File sizes saved\n");
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
	printf("Memory mapped\n");

	/**
	* 
	**/
	
	/* Everything till now was successful, begin actual process */
	compare_files(filearray);
	
	return 0;
}

int is_equal(void *value1, void *value2) {
	long *ar1 = (long*) value1;
	long *ar2 = (long*) value2;
	int maxiter = 4;
	printf("%d\n", maxiter);
	for(int index = 0; index < maxiter; index++) {
		printf("%08x == %08x\n", ar1[index], ar2[index]);
		if(ar1[index] != ar2[index])
			return index;
	}
	return -1;
}

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

void print(void* val) {
	int *value = (int*) val;
	printf("%08x %08x %08x %08x %08x %08x %08x %08x\n", value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7]);
}

void print_block(void *block) {
	printf("Printing block\n");
	char *blk = (char*) block;
	for(int i = 0; i < 32; i++)
		printf("%c", blk[i]);
	printf("\n");
}

char* blk_to_str(void *data) {	
	return (char*)data;
}

void compare_files(struct mmapfile *filearray) {
	/**
	* Todo: AVX-512 for string comparison
	**/
	/*
	__m512 str1, str2;
	str1 = _mm_load_ps(filearray[0].)
	*/
	printf("In compare_files\n");
	int equality;
	int idxs[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	__m256i mask;
	mask = _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1);	
	__m256i file1, file2;
	__m256i indices = _mm256_maskload_epi32(idxs, mask);
	__m256i zero = _mm256_setzero_si256();
	//print(&indices);
	__m256i mask7 = _mm256_set_epi32(8, 8, 8, 8, 8, 8, 8, 8);	
	__m256i stringarr[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
	struct blkdata file1blkdata, file2blkdata;

	//print(stringarr);
	/**
	* Todo: Break up mapped string into 256b blocks
	* and calculate the number of blocks required.
	**/
	numblks_req(filearray[0].fsize, &file1blkdata);
	numblks_req(filearray[1].fsize, &file2blkdata);
	printf("Number of blocks required: %d %d\n", file1blkdata.nblks, file2blkdata.nblks);

	__m256i blocksfile1[file1blkdata.nblks];
	__m256i blocksfile2[file2blkdata.nblks];
	int niter = file1blkdata.nblks < file2blkdata.nblks ? file2blkdata.nblks : file1blkdata.nblks;
	for(int blockno = 0; blockno < niter; blockno++) {
		blocksfile1[blockno] = _mm256_i32gather_epi32(filearray[0].memmptr, indices, 4);
		blocksfile2[blockno] = _mm256_i32gather_epi32(filearray[1].memmptr, indices, 4);
		indices = _mm256_add_epi32(indices, mask7);
		print(&indices);
	}
	__m256i result;
	result = _mm256_cmpeq_epi32(blocksfile1[0], blocksfile2[0]);
	print_block(&blocksfile1);
	print_block(&blocksfile2);
	print(&result);
	__m256i metaresult = _mm256_cmpeq_epi32(result, mask);
	print(&metaresult);	
	equality = is_equal(&result, &mask);
	if(equality == -1) {			
		/* Line seen in both files */
		printf("Line seen in both files\n");
		total[BOTH_FILES]++;
		fprintf(stdout, "%s", blk_to_str(&blocksfile1[0]));
	}
	else {
		/* Block mismatch, fallback to memcmp */
		equality = memcmp(&blocksfile1[0], &blocksfile2[0], 32);
		if(equality <= 0) {
			/* Line seen in first file */
			printf("Line seen in the first file\n");
			total[FIRST_FILE]++;
			print_block(&blocksfile1);
		}
		else {
			/* Line seen in second file */
			printf("Line seen in the second file\n");
			total[SECOND_FILE]++;
			print_block(&blocksfile2);
		}
	}
}
