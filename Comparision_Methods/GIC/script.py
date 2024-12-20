from main import GICM


if __name__ == '__main__':
    print("\nt2d Results:")
    GICM('t2d_gene_homo_betwenness.pt')
    print("\npd Results:")
    GICM('pd_gene_homo_betwenness.pt') 
    print("\nhd Results:")
    GICM('hd_gene_homo_betwenness.pt')
    print("\nsch Results:")
    GICM('sch_gene_homo_betwenness.pt')