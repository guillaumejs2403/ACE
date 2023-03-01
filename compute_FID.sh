OUTPATH=$1
EXPPATH=$2
TEMPPATH=.temp
if [ $# -eq 3 ]; then
    TEMPPATH=$3
fi

mkdir -p ${TEMPPATH}/real
mkdir -p ${TEMPPATH}/cf

cp -r ${OUTPATH}/Results/${EXPPATH}/CC/CCF/CF ${TEMPPATH}/cf
mv ${TEMPPATH}/cf/CF/* ${TEMPPATH}/cf
rm -rf ${TEMPPATH}/cf/CF
cp -r ${OUTPATH}/Results/${EXPPATH}/IC/CCF/CF ${TEMPPATH}/cf
mv ${TEMPPATH}/cf/CF/* ${TEMPPATH}/cf
rm -rf ${TEMPPATH}/cf/CF

cp -r ${OUTPATH}/Original/Correct/* ${TEMPPATH}/real
cp -r ${OUTPATH}/Original/Incorrect/* ${TEMPPATH}/real

echo 'Computing FID'

python -m pytorch_fid ${TEMPPATH}/real ${TEMPPATH}/cf --device cuda:0

rm -rf ${TEMPPATH}
