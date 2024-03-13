import sys

assert len(sys.argv) == 4, len(sys.argv)

prof_hits_path = sys.argv[1]
counts_hits_path = sys.argv[2]

merged_hits_path = sys.argv[3]


def load_hits(hits_path):
	with open(hits_path) as f:
		return [line.rstrip().split() for line in f]

def process_hits(hits):
	# ignore peak index and motif index columns
	return [(hit[0], int(hit[1]), int(hit[2]), hit[3], hit[5]) for hit in hits]

def relabel_hits(hits, label):
	# the 1 is there because bed6 format requires a number in that column
	return { tuple(list(hit[:3]) + [hit[3] + "_" + label] + [1, hit[4]]) for hit in hits }


print("Loading hits.")

prof_hits = set(process_hits(load_hits(prof_hits_path)))
counts_hits = set(process_hits(load_hits(counts_hits_path)))


print("Merging hits.")

hits_in_both = prof_hits.intersection(counts_hits)

prof_unique_hits = prof_hits - counts_hits
counts_unique_hits = counts_hits - prof_hits

prof_unique_hits = relabel_hits(prof_unique_hits, "profile")
counts_unique_hits = relabel_hits(counts_unique_hits, "counts")
hits_in_both = relabel_hits(hits_in_both, "profile,counts")

merged_hits = hits_in_both.union(prof_unique_hits).union(counts_unique_hits)

merged_hits = sorted(list(merged_hits))


print("Writing merged hits to file.")

def format_bed_line(line_list):
	return "\t".join([str(thing) for thing in line_list])


with open(merged_hits_path, "w") as f:
	for hit in merged_hits:
		f.write(format_bed_line(hit) + "\n")

print("Done.")

