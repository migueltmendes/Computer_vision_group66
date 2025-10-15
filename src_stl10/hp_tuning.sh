FREEZE_FEATURES=(True False)
LOAD_BEST=(True False)
BOTTLENECK_LIST=(avg_8x8 max_8x8 global_avg global_max 2x2_conv 3x3_conv 5x5_conv linear)
LRS=(0.1 0.01 0.001 0.0001)
WDS=(0 0.0001 0.001 0.01)
EPOCHS=(20 30 50)

MAX_COUNT=$(( ${#BOTTLENECK_LIST[@]} * ${#LRS[@]} * ${#WDS[@]} * (1 + 2 * ${#EPOCHS[@]}) ))
COUNT=0

echo "Completed $COUNT/$MAX_COUNT configurations."

for BOTTLENECK in "${BOTTLENECK_LIST[@]}"; do
    for LR in "${LRS[@]}"; do
        for WD in "${WDS[@]}"; do
            echo "Running with bottleneck=$BOTTLENECK, freeze_features=False, lr=$LR, weight_decay=$WD, epochs=20, load_best=True"
            python3 src_stl10/main.py --bottleneck_type "$BOTTLENECK" --lr "$LR" --weight_decay "$WD" --epochs "20" --use_amp --load_best

            COUNT=$((COUNT + 1))
            echo "Completed $COUNT/$MAX_COUNT configurations."

            for EPOCH in "${EPOCHS[@]}"; do
                echo "Running with bottleneck=$BOTTLENECK, freeze_features=True, lr=$LR, weight_decay=$WD, epochs=$EPOCH, load_best=True"
                python3 src_stl10/main.py --bottleneck_type "$BOTTLENECK" --freeze_features --lr "$LR" --weight_decay "$WD" --epochs "$EPOCH" --use_amp --load_best

                COUNT=$((COUNT + 1))
                echo "Completed $COUNT/$MAX_COUNT configurations."
            done

            for EPOCH in "${EPOCHS[@]}"; do
                echo "Running with bottleneck=$BOTTLENECK, freeze_features=True, lr=$LR, weight_decay=$WD, epochs=$EPOCH, load_best=False"
                python3 src_stl10/main.py --bottleneck_type "$BOTTLENECK" --freeze_features --lr "$LR" --weight_decay "$WD" --epochs "$EPOCH" --use_amp

                COUNT=$((COUNT + 1))
                echo "Completed $COUNT/$MAX_COUNT configurations."
            done
        done
    done
done