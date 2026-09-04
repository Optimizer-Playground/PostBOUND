-- provide an overview of the current contents of the shared buffers.
-- for each relation (relname), the table to which it belongs (owner_relname) will be returned, as well as the number of
-- pages in the buffer (buffers) and the kind of pages that are buffered ('table' for normal data tables, 'primary' for primary
-- key index pages or 'secondary' for auxillary index pages)
CREATE OR REPLACE FUNCTION pg_buffercache_table_summary(nspname NAME DEFAULT 'public')
RETURNS TABLE (nspname NAME, relname NAME, owner_relname NAME, pagetype TEXT, buffers BIGINT)
AS $$
	WITH tab_idxs AS (
		SELECT cls.relname,
			cls.oid,
			owner_cls.relname AS owner_relname,
			owner_cls.oid AS owner_oid,
			CASE WHEN idx.indisprimary THEN 'primary' ELSE 'secondary' END AS indtype
		FROM pg_index idx
			JOIN pg_class cls ON idx.indexrelid = cls.oid
			JOIN pg_class owner_cls ON idx.indrelid = owner_cls.oid
		ORDER BY owner_relname, relname
	)
	SELECT nsp.nspname,
		cls.relname,
		CASE WHEN tab_idxs.owner_relname IS NULL THEN cls.relname ELSE tab_idxs.owner_relname END AS owner_relname,
		CASE WHEN tab_idxs.indtype IS NULL THEN 'table' ELSE tab_idxs.indtype END AS pagetype,
		COUNT(*) AS buffers
	FROM pg_buffercache buf
		JOIN pg_class cls ON (buf.relfilenode = pg_relation_filenode(cls.oid)
							AND buf.reldatabase IN (0,
													(SELECT oid FROM pg_database WHERE datname = CURRENT_DATABASE()))
							)
		JOIN pg_namespace nsp ON nsp.oid = cls.relnamespace
		LEFT JOIN tab_idxs ON cls.oid = tab_idxs.oid
	WHERE nsp.nspname = 'public'
	GROUP BY nsp.nspname, cls.relname, tab_idxs.owner_relname, tab_idxs.indtype
	ORDER BY buffers
$$ LANGUAGE SQL;
