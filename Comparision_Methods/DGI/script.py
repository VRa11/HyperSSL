from main import DGIM


if __name__ == '__main__':
    print("\nt2d Results:")
    DGIM('t2d_gene_homo_betwenness.pt')
    print("\npd Results:")
    DGIM('pd_gene_homo_betwenness.pt') 
    print("\nhd Results:")
    DGIM('hd_gene_homo_betwenness.pt')
    print("\nsch Results:")
    DGIM('sch_gene_homo_betwenness.pt')