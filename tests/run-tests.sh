echo "Running GNU comm"
echo "================"
echo "Size: 256B"
time  comm -123 --nocheck-order ../comm/test/1.txt ../comm/test/1.txt
echo "Size: 726kB"
time  comm -123 --nocheck-order ../comm/test/prideandprejudice.txt ../comm/test/prideandprejudice.txt
echo "Size: 6.5MB"
time  comm -123 --nocheck-order ../comm/test/big.txt ../comm/test/big.txt
echo "Size: 108.2B"
time  comm -123 --nocheck-order ../comm/test/large.txt ../comm/test/large.txt

echo "Running SIMD Implementation"
echo "==========================="
echo "Size: 256B"
time  ../build/pcomm ../comm/test/1.txt ../comm/test/1.txt
echo "Size: 726kB"
time  ../build/pcomm ../comm/test/prideandprejudice.txt ../comm/test/prideandprejudice.txt
echo "Size: 6.5MB"
time  ../build/pcomm ../comm/test/big.txt ../comm/test/big.txt
echo "Size: 108.2B"
time  ../build/pcomm ../comm/test/large.txt ../comm/test/large.txt

echo "Running Parallelized SIMD Implementation"
echo "========================================"
echo "Size: 256B"
time  ../build/pcomm_threaded ../comm/test/1.txt ../comm/test/1.txt
echo "Size: 726kB"
time  ../build/pcomm_threaded ../comm/test/prideandprejudice.txt ../comm/test/prideandprejudice.txt
echo "Size: 6.5MB"
time  ../build/pcomm_threaded ../comm/test/big.txt ../comm/test/big.txt
echo "Size: 108.2B"
time  ../build/pcomm_threaded ../comm/test/large.txt ../comm/test/large.txt