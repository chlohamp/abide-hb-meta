import os
import os.path as op

import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from atlasreader import atlasreader
from nilearn import connectome, masking
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from wordcloud import WordCloud


def wordcloud(csv):
    df = pd.read_csv(csv)
    df = df.sort_values(by=["correlation"], ascending=False)
    df = df.head(20)
    term_freq_dict = {}
    for i, row in df.iterrows():
        print(row["feature"])
        print(row["correlation"])
        term_freq_dict[row["feature"]] = row["correlation"]

    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130**2
    mask = 255 * mask.astype(int)
    wc = WordCloud(background_color="white", mask=mask)
    wordcloud = wc.generate_from_frequencies(term_freq_dict)
    # Display the generated image:
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("{}.jpg".format(csv.strip(".csv")))
    plt.close()


def decode(img_decode):
    imgs_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/feature_maps"
    terms_df = pd.read_csv(op.join(imgs_dir, "characterized-ns-terms.csv"), sep=",")
    terms_decode = terms_df["FEATURE"].loc[
        terms_df["Classification"].isin(["Functional", "Clinical"])
    ]
    correlation = connectome.ConnectivityMeasure(kind="correlation")
    term_list = []
    correlation_list = []
    for term in terms_decode:
        print(term)
        if op.isfile(op.join(imgs_dir, "{}_association-test_z.nii.gz".format(term))):
            term_list.append(term)
            term_img = nib.load(op.join(imgs_dir, "{}_association-test_z.nii.gz".format(term)))
            term_mask = masking.compute_background_mask(term_img)
            correlation_list.append(
                correlation.fit_transform(
                    [np.transpose(masking.apply_mask([nib.load(img_decode), term_img], term_mask))]
                )[0][0, 1]
            )
    decode_df = pd.DataFrame()
    decode_df["feature"] = term_list
    decode_df["correlation"] = correlation_list
    decode_df.to_csv("{}.csv".format(img_decode.split(".nii.gz")[0]), index=False)


def get_peaks(img_file, output_dir):
    # get cluster + peak information from image
    stat_img = nib.load(img_file)
    if np.nonzero(stat_img.get_fdata())[0].any():
        atlas = ["aal"]
        voxel_thresh = np.min(stat_img.get_fdata()[np.nonzero(stat_img.get_fdata())])
        direction = "pos"
        cluster_extent = 1
        prob_thresh = 5
        min_distance = 15
        out_fn = op.join(
            output_dir, "{0}_clusterinfo.csv".format(op.basename(img_file).strip(".nii.gz"))
        )

        _, peaks_frame = atlasreader.get_statmap_info(
            stat_img,
            atlas=atlas,
            voxel_thresh=voxel_thresh,
            direction=direction,
            cluster_extent=cluster_extent,
            prob_thresh=prob_thresh,
            min_distance=min_distance,
        )

        # Only proceed if peaks_frame is not empty
        if not peaks_frame.empty:
            hemis = []
            labels = []
            for i, row in peaks_frame.iterrows():
                tmplabel = row["aal"]
                if tmplabel.split("_")[-1] in ["L", "R"]:
                    hemis.append(tmplabel.split("_")[-1])
                    labels.append(" ".join(tmplabel.split("_")[:-1]))
                else:
                    hemis.append("")
                    labels.append(" ".join(tmplabel.split("_")))

            peaks_frame["Hemisphere"] = hemis
            peaks_frame["Label"] = labels
            peaks_frame = peaks_frame.drop(columns=["aal"])
            peaks_frame = peaks_frame.rename(
                columns={
                    "cluster_id": "Cluster",
                    "peak_x": "x",
                    "peak_y": "y",
                    "peak_z": "z",
                    "peak_value": "Value",
                    "volume_mm": "Volume (mm^3)",
                }
            )

            # write output .csv files
            print(out_fn)
            peaks_frame.to_csv(out_fn, index=False, float_format="%5g")

            return peaks_frame
        else:
            print(f"No peaks found for {img_file}")
            return None



# def thresh_img(logp_img, z_img, p):
#     sig_inds = np.where(logp_img.get_fdata() > -np.log10(p))
#     z_img_data = z_img.get_fdata()
#     z_img_thresh_data = np.zeros(z_img.shape)
#     z_img_thresh_data[sig_inds] = z_img_data[sig_inds]
#     z_img = nib.Nifti1Image(z_img_thresh_data, z_img.affine)
#     return z_img

def thresh_img(img, threshold):
    """
    Threshold an image based on absolute value.
    Keeps voxels where |value| > threshold.
    """
    data = img.get_fdata()
    thresh_data = np.zeros_like(data)
    thresh_data[np.abs(data) > threshold] = data[np.abs(data) > threshold]
    return nib.Nifti1Image(thresh_data, img.affine)



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    """dendrogram(linkage_matrix, color_threshold=linkage_matrix[7, 2], **kwargs)"""

    j = 5
    set_link_color_palette(
        [matplotlib.colors.rgb2hex(cm.nipy_spectral(float(i) / j)) for i in range(j)]
    )
    dendrogram(linkage_matrix, **kwargs)
