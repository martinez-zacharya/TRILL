def setup(subparsers):
    visualize = subparsers.add_parser("visualize", help="Reduce dimensionality of embeddings to 2D")

    visualize.add_argument(
        "embeddings",
        help="Embeddings to be visualized",
        action="store"
    )

    visualize.add_argument(
        "--method",
        help="Method for reducing dimensions of embeddings. Default is PCA",
        action="store",
        choices=("PCA", "UMAP", "tSNE"),
        default="PCA"
    )
    visualize.add_argument(
        "--key",
        help="Input a CSV, with your group mappings for your embeddings where the first column is the label and the "
             "second column is the group to be colored.",
        action="store",
        default=False
    )


def run(args):
    import os

    import bokeh

    from trill.utils.visualize import reduce_dims, viz

    reduced_df, incsv = reduce_dims(args.name, args.embeddings, args.method)
    layout = viz(reduced_df, args)
    bokeh.io.output_file(filename=os.path.join(args.outdir, f"{args.name}_{args.method}_{incsv}.html"), title=args.name)
    bokeh.io.save(layout, filename=os.path.join(args.outdir, f"{args.name}_{args.method}_{incsv}.html"),
                  title=args.name)
