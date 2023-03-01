OUTPATH=$1
EXPPATH=$2
TEMPPATH=.temp
if [ $# -eq 3 ]; then
    TEMPPATH=$3
fi

mkdir -p ${TEMPPATH}/real/all
mkdir -p ${TEMPPATH}/real/split1
mkdir -p ${TEMPPATH}/real/split2
mkdir -p ${TEMPPATH}/cf/all
mkdir -p ${TEMPPATH}/cf/split1
mkdir -p ${TEMPPATH}/cf/split2

cp -r ${OUTPATH}/Results/${EXPPATH}/CC/CCF/CF ${TEMPPATH}/cf
mv ${TEMPPATH}/cf/CF/* ${TEMPPATH}/cf/all
rm -rf ${TEMPPATH}/cf/CF
cp -r ${OUTPATH}/Results/${EXPPATH}/IC/CCF/CF ${TEMPPATH}/cf
mv ${TEMPPATH}/cf/CF/* ${TEMPPATH}/cf/all
rm -rf ${TEMPPATH}/cf/CF

cp -r ${OUTPATH}/Original/Correct/* ${TEMPPATH}/real/all
cp -r ${OUTPATH}/Original/Incorrect/* ${TEMPPATH}/real/all

TOTALINSTANCES=$(ls ${TEMPPATH}/real/all | wc -l)

ls ${TEMPPATH}/real/all | shuf > ${TEMPPATH}/instance_list.txt
cat ${TEMPPATH}/instance_list.txt | tail -$(( ${TOTALINSTANCES} / 2 )) > ${TEMPPATH}/split1.txt
cat ${TEMPPATH}/instance_list.txt | head -$(( ${TOTALINSTANCES} / 2 )) > ${TEMPPATH}/split2.txt


for SPLIT in split1 split2;
do
    rsync --files-from=${TEMPPATH}/${SPLIT}.txt ${TEMPPATH}/real/all/${FILE} ${TEMPPATH}/real/${SPLIT}
    rsync --files-from=${TEMPPATH}/${SPLIT}.txt ${TEMPPATH}/cf/all/${FILE} ${TEMPPATH}/cf/${SPLIT}
    # for FILE in $(cat ${TEMPPATH}/${SPLIT}.txt);
    # do
        # mv ${TEMPPATH}/real/all/${FILE} ${TEMPPATH}/real/${SPLIT}
        # mv ${TEMPPATH}/cf/all/${FILE} ${TEMPPATH}/cf/${SPLIT}
    # done
done

python -m pytorch_fid ${TEMPPATH}/real/split1 ${TEMPPATH}/cf/split2 --device cuda:0
python -m pytorch_fid ${TEMPPATH}/real/split2 ${TEMPPATH}/cf/split1 --device cuda:0
# python -m pytorch_fid ${TEMPPATH}/real/split2 ${TEMPPATH}/real/split1 --device cuda:0

rm -rf ${TEMPPATH}

