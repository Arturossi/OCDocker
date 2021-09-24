from pymol import cmd,stored

set depth_cue, 1
set fog_start, 0.4

set_color b_col, [36,36,85]
set_color t_col, [10,10,10]
set bg_rgb_bottom, b_col
set bg_rgb_top, t_col      
set bg_gradient

set  spec_power  =  200
set  spec_refl   =  0

load "data/rec.crg.pdb", protein
create ligands, protein and organic
select xlig, protein and organic
delete xlig

hide everything, all

color white, elem c
color bluewhite, protein
#show_as cartoon, protein
show surface, protein
#set transparency, 0.15

show sticks, ligands
set stick_color, magenta

load "data/rec.crg.pdb_points.pdb.gz", points
hide nonbonded, points
show nb_spheres, points
set sphere_scale, 0.2, points
cmd.spectrum("b", "green_red", selection="points", minimum=0, maximum=0.7)


stored.list=[]
cmd.iterate("(resn STP)","stored.list.append(resi)")    # read info about residues STP
lastSTP=stored.list[-1] # get the index of the last residue
hide lines, resn STP

cmd.select("rest", "resn STP and resi 0")

for my_index in range(1,int(lastSTP)+1): cmd.select("pocket"+str(my_index), "resn STP and resi "+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.show("spheres","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_scale","0.4","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_transparency","0.1","pocket"+str(my_index))



set_color pcol1 = [0.361,0.576,0.902]
select surf_pocket1, protein and id [418,597,598,593,447,450,574,566,472,473,1167,469,1165,599,446,439,1961,449,1905,1907,1930,1908,1909,475,477,1701,1725,607,1175,1177,1172,1173,1174,1176,1171,1178,1182,1183,1895,1896,1868,1952,1929,1931,1723,1724,1758,1759,1184,1186,1734,1898,1847,1251] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.278,0.341,0.702]
select surf_pocket2, protein and id [1299,1300,1741,1742,1765,1766,1762,1764,1801,1763,1799,1800,1769,1798,1806,1772,1771,1774,1775,1776,1767,1773,1810,1305,1306,1307,1309,1311,1308,1310,1803,1804,1805,1268,1802,1337] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.424,0.361,0.902]
select surf_pocket3, protein and id [662,689,691,692,693,695,696,714,717,848,851,852,853,756,842,844,716,666,668,906,908,876,907,935,322,324,963] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.435,0.278,0.702]
select surf_pocket4, protein and id [331,279,291,292,293,2078,2064,2040,2079,2077,300,301,2109,262,250,2095,1618,1590,1617,1586,1610,732,734] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.698,0.361,0.902]
select surf_pocket5, protein and id [590,979,980,981,943,945,946,947,948,941,592,560,561,562,940,944,942,361,388,413,622,360,362] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.651,0.278,0.702]
select surf_pocket6, protein and id [119,140,145,146,141,171,2008,2009,2010,2019,2021,2020,2022,2023,2048,2142,2145,2114,2116,2149,2154,2184,120,117,121] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.902,0.361,0.824]
select surf_pocket7, protein and id [1410,1623,1385,1388,1390,1407,1450,1379,1384,1355,1622,1644,1645,1643,1674,1647,1648] 
set surface_color,  pcol7, surf_pocket7 
set_color pcol8 = [0.702,0.278,0.533]
select surf_pocket8, protein and id [163,383,160,183,184,162,359,362,181,185,186,189,182,190,191,193,194,215,216,217,218,219,220,221,222,223,192,195,196] 
set surface_color,  pcol8, surf_pocket8 
set_color pcol9 = [0.902,0.361,0.545]
select surf_pocket9, protein and id [215,217,182,338,209,183,184,362,336,337,303,305,308] 
set surface_color,  pcol9, surf_pocket9 
set_color pcol10 = [0.702,0.278,0.318]
select surf_pocket10, protein and id [638,1320,994,668,669,1350,1318,1322,963,964,1315] 
set surface_color,  pcol10, surf_pocket10 
set_color pcol11 = [0.902,0.451,0.361]
select surf_pocket11, protein and id [374,395,1983,617,618,599,650,623,624,1700] 
set surface_color,  pcol11, surf_pocket11 
set_color pcol12 = [0.702,0.459,0.278]
select surf_pocket12, protein and id [1771,1774,1775,1776,1770,1773,1836,1777,1750,1890,1749,1849,1853,1850,1860,1782] 
set surface_color,  pcol12, surf_pocket12 
set_color pcol13 = [0.902,0.729,0.361]
select surf_pocket13, protein and id [661,670,671,675,914,935,910,934,344,915,662] 
set surface_color,  pcol13, surf_pocket13 
set_color pcol14 = [0.702,0.675,0.278]
select surf_pocket14, protein and id [1658,1682,1688,1945,1996,1969] 
set surface_color,  pcol14, surf_pocket14 


deselect

orient
