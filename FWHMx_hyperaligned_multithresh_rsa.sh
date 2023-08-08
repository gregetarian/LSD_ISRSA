#!/bin/bash 
#code to perform cluster-size based FDR correction. 
#This will compute the spatial autocorrelation function of the original timeseries data using  3dFWHMx in AFNI (which isn't quite optimal - see their docs)
#against which noise is simulated using 3dClustsim for computation of optimal cluster-sizes vs alpha thresholds of your chosing. Here we use 0.05.

songs="1 2"
dim="EgoDissolution Simple Cmplx"
#dim="Simple"
#dim="Cmplx"
wdir="/Users/gcooper/Downloads/FD_040_NoScrub/LSD/rest2/hyperaligned"


mkdir "$wdir"/results/CScorrect

for d in $dim; do
    for song in $songs; do
        echo "$song"
        echo "$song"
        echo "$song"
        echo "$song"
        echo "$song"
        
        mkdir -p "$wdir"/results/CScorrect/clustsim_"$song"_"$d"


        #z-score maps
        MEAN=`3dBrickStat -mean "$wdir"/results/isrsa_brain_song"$song"_"$d".nii.gz`
        STD=`3dBrickStat -stdev "$wdir"/results/isrsa_brain_song"$song"_"$d".nii.gz`

        rm "$wdir"/results/CScorrect/Z_isrsa_brain_song"$song"_"$d".nii.gz

        3dcalc -prefix "$wdir"/results/CScorrect/Z_isrsa_brain_song"$song"_"$d".nii.gz \
        -a "$wdir"/results/isrsa_brain_song"$song"_"$d".nii.gz \
        -expr "(a/a)*((a-$MEAN)/$STD)"


        if [[ $song == 1 ]]; then
            perps="S01 S04 S09 S13 S17 S20"
        elif [[ $song == 2 ]]; then 
            perps="S02 S06 S10 S11 S18 S19"
        fi

        acf1=0
        acf2=0
        acf3=0
        for perp in $perps; do
            #estimate noise from z-scored maps [probably incorrect, needs some attention]
            acf=`3dFWHMx -ACF NULL -2difMAD -input "$wdir"/"$perp"_hyperaligned.nii.gz`
            
            #extract within-participant ACF params
            perp_acf1=`echo $acf | awk '{print $18}'`
            perp_acf2=`echo $acf | awk '{print $19}'`
            perp_acf3=`echo $acf | awk '{print $20}'`


            #update cumulative ACF params
            acf1=`echo "$perp_acf2 + $acf1" | bc`
            acf2=`echo "$perp_acf1 + $acf1" | bc`
            acf2=`echo "$perp_acf1 + $acf1" | bc`

            echo $acf1 $acf2 $acf3
        done

        #calculate average ACF params 
        num_participants=`printf '%s\n' "${perps[@]}" | wc -w`

        acf1=`echo "$acf1 / $num_participants" | bc`
        acf2=`echo "$acf2 / $num_participants" | bc`
        acf3=`echo "$acf3 / $num_participants" | bc`

        #run clustsim
        3dClustSim -acf $acf1 $acf2 $acf3 \
        -nodec -both -iter 10001 \
        -pthr 0.05 0.02 0.01 0.005 0.002 0.001 \
        -athr 0.05 0.02 0.01 0.005 0.002 0.001 \
        -prefix "$wdir"/results/CScorrect/clustsim_"$song"_"$d"/clustsim_"$song"_"$d"

        pthresholds="05 02 01 005 002 001"
        cluster_table="$wdir/clustsim_"$song"_"$d"/clustsim_"$song"_"$d".NN2_2sided.1D"
        athreshold_col="2"

        mkdir "$wdir"/results/CScorrect/LME_thresh_"$song"_"$d"_a05


        #cluster-size correction 
        for pthreshold in $pthresholds; do
            if [ "$pthreshold" = "05" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 9 {print $var}'`
                    thresh=1.96
                    echo $cs
            fi
            if [ "$pthreshold" = "02" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 10 {print $var}'`
                    thresh=2.33
                    echo $cs
            fi
            if [ "$pthreshold" = "01" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 11 {print $var}'`
                    thresh=2.58
                    echo $cs
            fi
            if [ "$pthreshold" = "005" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 12 {print $var}'`
                    thresh=2.81
                    echo $cs
            fi
            if [ "$pthreshold" = "002" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 13 {print $var}'`
                    thresh=3.09
                    echo $cs
            fi
            if [ "$pthreshold" = "001" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 14 {print $var}'`
                    thresh=3.29
                    echo $cs
            fi
            if [ "$pthreshold" = "0005" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 15 {print $var}'`
                    thresh=3.48
                    echo $cs
            fi
            if [ "$pthreshold" = "0002" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 16 {print $var}'`
                    thresh=3.72
                    echo $cs
            fi
            if [ "$pthreshold" = "0001" ]
            then
                    cs=`cat $cluster_table | awk -v var="$athreshold_col" 'NR == 17 {print $var}'`
                    thresh=3.89
                    echo $cs
            fi

            3dmerge \
            -dxyz=1 -1clust 1 "$cs" -2clip -100000000 "$thresh" \
            -prefix "$wdir"/results/CScorrect/LME_thresh_"$song"_"$d"_a05/cs"$cs"_t"$thresh".nii.gz \
            "$wdir"/results/CScorrect/Z_isrsa_brain_song"$song"_"$d".nii.gz
        done


        
        3dmerge \
        -nozero \
        -gnzmean \
        -prefix "$wdir"/results/CScorrect/"$song"_"$d"_cs_all_thresh_all_a05.nii.gz \
        "$wdir"/results/CScorrect/LME_thresh_"$song"_"$d"_a05/cs*_t*.nii.gz

    done

    #calculate overlapping clusters between songs
    rm "$wdir"/results/CScorrect/"$d"_overlaps_cthresh_S1S2_a05.nii.gz
    3dcalc -prefix "$wdir"/results/CScorrect/"$d"_overlaps_cthresh_S1S2_a05.nii.gz \
    -a "$wdir"/results/CScorrect/1_"$d"_cs_all_thresh_all_a05.nii.gz \
    -b "$wdir"/results/CScorrect/2_"$d"_cs_all_thresh_all_a05.nii.gz \
    -expr 'step(a)+(2*step(b))'
done











