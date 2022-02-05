import osmium as osm
import pandas as pd

class OSMHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.osm_data = []

    def tag_inventory(self, elem, elem_type):
        for tag in elem.tags:
            if elem_type=='node':
                self.osm_data.append([elem_type,
                                       elem.id,
                                       elem.version,
                                       elem.visible,
                                       pd.Timestamp(elem.timestamp),
                                       elem.uid,
                                       elem.changeset,
                                       len(elem.tags),
                                       tag.k,
                                       tag.v,
                                       elem.location.lat,
                                       elem.location.lon])
            else:
                self.osm_data.append([elem_type,
                                       elem.id,
                                       elem.version,
                                       elem.visible,
                                       pd.Timestamp(elem.timestamp),
                                       elem.uid,
                                       elem.changeset,
                                       len(elem.tags),
                                       tag.k,
                                       tag.v,
                                       0,
                                       0])

    def node(self, n):
        self.tag_inventory(n, "node")

    def way(self, w):
        self.tag_inventory(w, "way")

    def relation(self, r):
        self.tag_inventory(r, "relation")


osmhandler = OSMHandler()
# scan the input file and fills the handler list accordingly
osmhandler.apply_file("data/nuernberg.osm")

# transform the list into a pandas DataFrame
data_colnames = ['type', 'id', 'version', 'visible', 'ts', 'uid',
                 'chgset', 'ntags', 'tagkey', 'tagvalue',
                 'lat', 'lon']
df_osm = pd.DataFrame(osmhandler.osm_data, columns=data_colnames)

df_osm.tagkey.value_counts().head(50)
amenities = df_osm[df_osm.tagkey=='amenity']
amenities.to_parquet('data/amenities.parquet')
